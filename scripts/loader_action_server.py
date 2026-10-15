#!/usr/bin/env python3
import time
import threading
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from rclpy.action.server import ServerGoalHandle

from yoko_loader_tester.action import Trigger
from std_msgs.msg import Bool, Int32, Int16, Float64

from enum import Enum, auto

def pulse_bool(publisher):
    msg = Bool()
    msg.data = True
    publisher.publish(msg)

    time.sleep(0.2)

    msg.data = False
    publisher.publish(msg)

class VehicleLoadingState(Enum):
    InitialPos  = auto()
    PushToLimit = auto()
    Scooping    = auto()
    Lifting     = auto()
    Fallback    = auto()

class SequenceExecutor(Node):

    def __init__(self):
        super().__init__("load_unload_action_server")

        self.armController      = ArmPLCController(self)
        self.vehicleControlller = VehicleMotionController(
            node    =  self,
            kp      =  0.25,
            ki      =  0.01,
            kd      =  0.01,
            v_nominal = 4.0,
            max_cmd =  3.0,
            min_cmd = -3.0,
            max_i   =  10.0,
            min_i   = -10.0 
        )
        
        self.pub_enable_arm_ctrl        = self.create_publisher(Bool,    '/arm/enable_ctrl',           10)
        self.pub_enable_arm_arm_ctrl    = self.create_publisher(Bool,    '/arm/enable_arm_ctrl',       10)
        self.pub_enable_arm_bucket_ctrl = self.create_publisher(Bool,    '/arm/enable_bucket_ctrl',    10)
        self.pub_arm_target_pos         = self.create_publisher(Float64, '/arm/set_arm_target_pos',    10)
        self.pub_bucket_target_pos      = self.create_publisher(Float64, '/arm/set_bucket_target_pos', 10)
        self.pub_arm_move               = self.create_publisher(Bool,    '/arm/arm_move_to_pos',       10)
        self.pub_bucket_move            = self.create_publisher(Bool,    '/arm/bucket_move_to_pos',    10)
        self.pub_linear_spd             = self.create_publisher(Float64, '/vehicle/set_target_spd',    10)
        self.pub_linear_mov             = self.create_publisher(Bool,    '/vehicle/move',              10)
        self.pub_linear_stop            = self.create_publisher(Bool,    '/vehicle/stop_move',         10)

        self.create_subscription(Float64, "/vehicle/speed",                lambda msg : self.vehicle_speed_callback(msg),  10)
        self.create_subscription(Bool,    "/arm/is_arm_reached_target",    lambda msg : self.arm_reached_callback(msg),    10)
        self.create_subscription(Bool,    "/arm/is_bucket_reached_target", lambda msg : self.bucket_reached_callback(msg), 10)
        self.create_subscription(Float64, "/vehicle/motor_torque",         lambda msg : self.vehicle_motor_torque_callback(msg), 10)

        self.vehicle_speed_callback  = self.vehicleControlller.vehicle_speed_callback
        self.arm_reached_callback    = self.armController.arm_reached_callback
        self.bucket_reached_callback = self.armController.bucket_reached_callback 

        self.get_logger().info("Initializing action server")
        self.action_server = ActionServer(
            node             = self,
            action_type      = Trigger,
            action_name      = "/arm_loader/execute_sequence",
            execute_callback = self.execute_loader_goal,
        )
        self.get_logger().info("Initialize action server successfully")

        self.targets = [
            [0.0, -0.4, None],
            [0.0, None, 0.25],
            [5.0, None, None],
            [5.0, None, 0.0],
            [0.0, 0.7, None],
            [-5.0, None, None],
        ]

        self.LOADING_SEQ = [
            VehicleLoadingState.InitialPos,
            VehicleLoadingState.PushToLimit,
            VehicleLoadingState.Scooping,
            VehicleLoadingState.Lifting,
            VehicleLoadingState.Fallback
        ]

        self.state_reset()

    def vehicle_speed_callback(self, msg:Float64):
        raise NotImplementedError

    def arm_reached_callback(self, msg:Bool):
        raise NotImplementedError

    def bucket_reached_callback(self, msg:Bool):
        raise NotImplementedError
    
    def state_reset(self):
        self.loader_traj_point_no = -1
        self.loader_traj_is_done  = True
        self.is_cancled = False
        self.get_logger().info("Sequence action server state resetted")
        
    def send_vehicle_goal(self, target_pos:float):
        return self.vehicleControlller.control_pos(target_pos)

    def execute_sequence(self):

        ret = True
        self.loader_traj_point_no = 0
        self.loader_traj_is_done = False

        for idx, seq in enumerate(self.LOADING_SEQ):
            time.sleep(0.5)

            self.loader_traj_point_no = idx

            if self.is_cancled:
                self.get_logger().info(f"Execute loading sequence cancled")
                ret = False
                break
            
            if seq == VehicleLoadingState.InitialPos:
                ret = self.init_pos_procedure()

            elif seq == VehicleLoadingState.PushToLimit:
                ret = self.push_to_limit_procedure()

            elif seq == VehicleLoadingState.Scooping:
                ret = self.scooping_procedure()

            elif seq == VehicleLoadingState.Lifting:
                ret = self.lifting_procedure()

            elif seq == VehicleLoadingState.Fallback:
                ret = self.fallback_procedure()

            if not ret:
                break

        if ret:
            self.get_logger().info("All targets completed.")
        else:
            self.get_logger().info(f"Loading sequence incompleted at state[{self.loader_traj_point_no}]")

        self.state_reset()

    def init_pos_procedure(self, j1_pos:float=-0.4, j2_pos:float=0.25):
            
        self.get_logger().info(f"Execute Initial arm loading position J1=[{j1_pos}], J2=[{j2_pos}]")

        if not self.armController.send_loader_goal(j1_pos, None):
            self.get_logger().info(f"Fail execute initial scooping pos [J1_POS={j1_pos}]")
            return False

        if not self.armController.send_loader_goal(None, j2_pos):
            self.get_logger().info(f"Fail execute initial scooping pos [J2_POS={j2_pos}]")
            return False
        
        self.get_logger().info("Initial loading procedure completed successfully.")

    def push_to_limit_procedure(self, tor_thresh:float=1500.0):
        self.get_logger().info(f"Execute push_to_limit_procedure tor_thresh=[{tor_thresh}]")
        ret = self.vehicleControlller.push_until_torque_limit(tor_thresh=tor_thresh)
        if ret:
            self.get_logger().info("push_to_limit procedure completed successfully.")
        else:
            self.get_logger().info("push_to_limit procedure fail.")
        return ret

    def scooping_procedure(self, j2_pos:float=0.0):
        self.get_logger().info(f"Execute scooping procedure, J2_POS=[{j2_pos}]")

        if not self.armController.send_loader_goal(None, j2_pos):
            self.get_logger().info(f"Fail execute scooping pos [J2_POS={j2_pos}]")
            return False

        self.get_logger().info("scooping procedure completed successfully.")
        return True

    def lifting_procedure(self, j1_pos:float=0.0):
        self.get_logger().info(f"Execute lifting procedure, J1_POS=[{j1_pos}]")

        if not self.armController.send_loader_goal(j1_pos, None):
            self.get_logger().info(f"Fail execute lifting pos [J1_POS={j1_pos}]")
            return False

        self.get_logger().info("lifting procedure completed successfully.")
        return True

    def fallback_procedure(self, delay_time:float=20.0):
        self.get_logger().info(f"Execute fallback procedure, delay_time=[{delay_time}]")
        ret = self.vehicleControlller.move_forward_w_delay(delay_time)
        if ret:
            self.get_logger().info("fallback procedure completed successfully.")
        else:
            self.get_logger().info("fallback procedure fail.")
        return ret

    def execute_loader_goal(self, goal_handle: ServerGoalHandle):
        self.get_logger().info("Received request to execute loader sequence.")

        def process():
            feedback = Trigger.Feedback()
            result = Trigger.Result()
    
            self.loader_traj_is_done  = False
            try:
                self.get_logger().info("Execute action is processing")
                while rclpy.ok():
                    if goal_handle.is_cancel_requested:
                        self.get_logger().info("Loader action canceled.")
                        self.sequence_cancel_callback()
                        goal_handle.canceled()
                        result.success = False
                        return result
    
                    is_done, is_cancled ,state_num = self.get_sequence_state()
    
                    feedback.state_num = state_num  
                    goal_handle.publish_feedback(feedback)
    
                    if is_cancled:
                        self.get_logger().error("Loader sequence is canceled.")
                        goal_handle.abort()
                        result.success = False
                        self.state_reset()
                        return result
    
                    if is_done:
                        self.get_logger().info("Loader sequence completed successfully.")
                        break

                    rclpy.spin_once(self, timeout_sec=0.01)
    
            except Exception as e:
                self.get_logger().error(f"Error during sequence execution: {e}")
                goal_handle.abort()
                result.success = False
                return result
    
            goal_handle.succeed()
            result.success = True
            return result

        execute_process = threading.Thread(
            target = lambda : setattr(
                self,
                "_execute_result",
                process()
            )
        )
        execute_process.deamon = True
        execute_process.start()
        
        process_thread = threading.Thread(target=self.execute_sequence, daemon=True)
        process_thread.start()

        execute_process.join()
        process_thread.join() 

        return self._execute_result

    def sequence_cancel_callback(self):
        self.is_cancled = True
        self.vehicleControlller.is_cancled = True

    def get_sequence_state(self):
        return self.loader_traj_is_done, self.is_cancled, self.loader_traj_point_no

    def vehicle_motor_torque_callback(self, msg:Float64):
        self.vehicleControlller.vehicle_tor = msg.data

class ArmPLCController:
    def __init__(self, node:SequenceExecutor):
        self.node = node

        self.is_arm_reached = False
        self.is_bucket_reached = False

    def set_enable_arm_ctrl(self):
        msg = Bool()
        msg.data = True
        self.node.pub_enable_arm_ctrl.publish(msg)

    def set_disable_arm_ctrl(self):
        msg = Bool()
        msg.data = False
        self.node.pub_enable_arm_ctrl.publish(msg)

    def set_enable_arm_arm_ctrl(self):
        msg = Bool()
        msg.data = True
        self.node.pub_enable_arm_arm_ctrl.publish(msg)

    def set_disable_arm_arm_ctrl(self):
        msg = Bool()
        msg.data = False
        self.node.pub_enable_arm_arm_ctrl.publish(msg)

    def set_enable_arm_bucket_ctrl(self):
        msg = Bool()
        msg.data = True
        self.node.pub_enable_arm_bucket_ctrl.publish(msg)

    def set_disable_arm_bucket_ctrl(self):
        msg = Bool()
        msg.data = False
        self.node.pub_enable_arm_bucket_ctrl.publish(msg)

    def set_arm_target_pos(self, pos:float):
        msg = Float64()
        msg.data = pos
        # self.is_arm_reached = False
        self.node.pub_arm_target_pos.publish(msg)

    def set_bucket_target_pos(self, pos:float):
        msg = Float64()
        msg.data = pos
        # self.is_bucket_reached = False
        self.node.pub_bucket_target_pos.publish(msg)

    def set_arm_move(self):
        pulse_bool(self.node.pub_arm_move)

    def set_bucket_move(self):
        pulse_bool(self.node.pub_bucket_move)

    def wait_until_reached(self, is_reached:callable, timeout:float=30.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            
            time.sleep(0.01)

            if is_reached():
                return True
            
        return False
    
    def send_loader_goal(self, j1:float=None, j2:float=None):
        self.set_disable_arm_ctrl()
        self.set_enable_arm_ctrl()

        if j1 is not None:
            self.is_arm_reached = False
            self.set_disable_arm_bucket_ctrl()
            self.set_enable_arm_arm_ctrl()
            self.set_arm_target_pos(j1)
            self.set_arm_move()
        if j2 is not None:
            self.is_bucket_reached = False
            self.set_disable_arm_arm_ctrl()
            self.set_enable_arm_bucket_ctrl()
            self.set_bucket_target_pos(j2)
            self.set_bucket_move()

        print(f"arm-bucket1 {self.is_arm_reached}-{self.is_bucket_reached}")
        j1_ret = True
        if j1 is not None:
            j1_ret = self.wait_until_reached(lambda : self.is_arm_reached)
        print(f"arm-bucket2 {self.is_arm_reached}-{self.is_bucket_reached}")
        j2_ret = True
        if j2 is not None:
            j2_ret = self.wait_until_reached(lambda : self.is_bucket_reached)
        print(f"arm-bucket3 {self.is_arm_reached}-{self.is_bucket_reached}")
        print(j1_ret, j2_ret)
        return j1_ret and j2_ret

    def arm_reached_callback(self, msg:Bool):
        if msg.data:
            self.is_arm_reached = True
            print(f"arm msg {msg.data}-{self.is_arm_reached}")
            self.node.get_logger().info("Arm reached target position")

    def bucket_reached_callback(self, msg:Bool):
        if msg.data:
            self.is_bucket_reached = True
            print(f"bucket msg {msg.data}-{self.is_bucket_reached}")
            self.node.get_logger().info("Bucket reached target position")

class VehicleMotionController:
    def __init__(self, node:SequenceExecutor, kp:float, ki:float, kd:float, v_nominal:float, max_cmd:float, min_cmd:float, max_i:float, min_i:float, tol:float=0.2):
        self.node = node

        self._vehicle_spd  : float = 0.0
        self._vehicle_dist : float = 0.0

        self.reset()
        self.set_control_gain(kp, ki, kd)
        self.set_control_limit(v_nominal, max_cmd, min_cmd, max_i, min_i)
        self.tol = tol
        self.vehicle_tor = 0.0

    def set_control_gain(self, kp:float, ki:float, kd:float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_control_limit(self, v_nominal:float ,max_cmd:float, min_cmd:float, max_i:float, min_i:float):
        self.max_cmd = max_cmd
        self.min_cmd = min_cmd
        self.max_i   = max_i
        self.min_i   = min_i
        self.v_nominal = v_nominal

    def reset(self):
        self._vehicle_spd  = 0.0
        self._vehicle_dist = 0.0
        self.err           = 0.0
        self.err_i         = 0.0
        self.err_d         = 0.0
        self.err_prev      = 0.0
        self.is_cancled    = False

    def compute_vehicle_dist(self, dt):
        self._vehicle_dist += self._vehicle_spd * dt

    def vehicle_speed_callback(self, msg:Float64):
        self._vehicle_spd = msg.data

    def execute(self, target_pos:float):
        process = threading.Thread(target=self.control_pos, args=(target_pos,), daemon=True)
        process.start()
    
    def control_pos(self, target_pos:float, timeout:float=60.0):

        start_time = time.time()
        prev_time = start_time

        self.node._logger.info(f"VehicleMotionControl receive target position = [{target_pos}]")
        while not self.is_cancled:
            now_time = time.time()
            dt = now_time - prev_time
            prev_time = now_time

            time_elapse = now_time - start_time
            if time_elapse > timeout:
                self.is_cancled = True
                break

            v_cmd, err = self.control_law(target_pos = target_pos, dt = dt)
            self.node.get_logger().info(f"VehicleMotionControl, error = [{err}], command speed = [{v_cmd}]")
            self.set_linear_spd(v_cmd)

            self.compute_vehicle_dist(dt)
            if abs(err) < self.tol:
                self.set_linear_stop_mov()
                break

        ret = True
        if self.is_cancled:
            ret = False
        else:
            self.node._logger.info(f"VehicleMotionControl target position [{target_pos}] is finished")

        self.reset()
        return ret

    def control_law(self, 
                    target_pos : float, 
                    dt         : float,
                    ):
    
        self.err      = target_pos - self._vehicle_dist

        # self.err_i   += self.err * dt
        # self.err_i    = np.clip(self.err_i, self.min_i, self.max_i)

        # self.err_d    = (self.err - self.err_prev)/dt
        # self.err_prev = self.err

        # v_cmd         = self.kp * self.err + self.ki * self.err_i + self.kd * self.err_d
        v_cmd         = np.sign(self.err) * self.v_nominal
        v_cmd         = np.clip(v_cmd, self.min_cmd, self.max_cmd)

        return v_cmd, self.err

    def set_linear_spd(self, spd:float):
        msg_spd = Float64()
        msg_spd.data = spd
        self.node.pub_linear_spd.publish(msg_spd)
        pulse_bool(self.node.pub_linear_mov)

    def set_linear_stop_mov(self):
        pulse_bool(self.node.pub_linear_stop)

    def push_until_torque_limit(self, tor_thresh:float = 1500.0, timeout:float=60.0):

        start_time = time.time()
        prev_time = start_time

        self.node._logger.info(f"VehicleTorqueControl: torque thresh = {tor_thresh}")
        while not self.is_cancled:
            now_time = time.time()
            dt = now_time - prev_time
            prev_time = now_time

            time_elapse = now_time - start_time
            if time_elapse > timeout:
                self.is_cancled = True
                break

            v_cmd = self.v_nominal
            self.node.get_logger().info(f"VehicleTorqueControl: current torque = [{self.vehicle_tor}], command speed = [{v_cmd}]")
            self.set_linear_spd(v_cmd)

            if self.vehicle_tor >= tor_thresh:
                self.set_linear_stop_mov()
                break

        ret = True
        if self.is_cancled:
            ret = False
        else:
            self.node._logger.info(f"VehicleTorqueControl is finished")

        self.reset()
        return ret

    def move_forward_w_delay(self, delay_time:float=10.0):
        start_time = time.time()

        self.node._logger.info(f"VehicleMoveWDelay: delay_time = {delay_time}")
        while not self.is_cancled:

            now_time = time.time()
            time_elapse = now_time - start_time
            if time_elapse > delay_time:
                self.set_linear_stop_mov()
                break

            v_cmd = -self.v_nominal
            self.node.get_logger().info(f"VehicleMoveWDelay: delay_time = [{delay_time}], command speed = [{v_cmd}]")
            self.set_linear_spd(v_cmd)

        ret = True
        if self.is_cancled:
            ret = False
        else:
            self.node._logger.info(f"VehicleMoveWDelay is finished")

        self.reset()
        return ret

def main(args=None):
    rclpy.init(args=args)

    node = SequenceExecutor()
    node.get_logger().info("Wheel loader action server for load/unload start!!")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()