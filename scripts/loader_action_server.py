#!/usr/bin/env python3
import time
import threading
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from rclpy.action.server import ServerGoalHandle

from yoko_loader_tester.action import Trigger
from std_msgs.msg import Bool, Int32, Int16, Float64, Float64MultiArray

def pulse_bool(publisher):
    msg = Bool()
    msg.data = True
    publisher.publish(msg)

    time.sleep(0.1)

    msg.data = False
    publisher.publish(msg)
    
class SequenceExecutor(Node):

    def __init__(self):
        super().__init__("load_unload_action_server")

        self.armController      = ArmPLCController(self)
        self.vehicleControlller = VehicleMotionController(
            node    =  self,
            kp      =  1.0,
            ki      =  0.0,
            kd      =  0.0,
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
        self.create_subscription(Float64MultiArray, "/loader_target_position", self.loader_target_pos_callback, 10)

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
            [0.24197153387468573, 0.0, 0.0],
            [1.0, 0.21979246164780855, 0.3033429986776818],
            [2.0, 0.22009244819553492, 0.3037867161184603],
            [2.0, 0.22009244819553492, 0.3037867161184603],
            [0.0, 0.8029340032443635, 0.15712545046230594]
        ]

        self.state_reset()

    def vehicle_speed_callback(self, msg:Float64):
        raise NotImplementedError

    def arm_reached_callback(self, msg:Bool):
        raise NotImplementedError

    def bucket_reached_callback(self, msg:Bool):
        raise NotImplementedError

    def loader_target_pos_callback(self, msg:Float64MultiArray):
        self.targets = np.array(msg.data).reshape(-1,3)
    
    def state_reset(self):
        self.loader_traj_point_no = -1
        self.loader_traj_is_done  = True
        self.is_cancled = False
        self.get_logger().info("Sequence action server state resetted")
        
    def send_vehicle_goal(self, target_pos:float):
        return self.vehicleControlller.control_pos(target_pos)

    def execute_sequence(self):

        for idx, (x, j1, j2) in enumerate(self.targets):

            if self.is_cancled:
                break
            
            self.get_logger().info(f"Execute Target {idx+1}/{len(self.targets)}")
            self.loader_traj_point_no = idx
            self.loader_traj_is_done = False

            vehicle_success = False
            loader_success = False

            vehicle_thread = threading.Thread(
                target=lambda: setattr(
                    self,
                    "_vehicle_result",
                    self.send_vehicle_goal(x)
                )
            )

            loader_thread = threading.Thread(
                target=lambda: setattr(
                    self,
                    "_loader_result",
                    self.armController.send_loader_goal(j1, j2)
                )
            )

            vehicle_thread.start()
            loader_thread.start()

            vehicle_thread.join()
            loader_thread.join()

            vehicle_success = self._vehicle_result
            loader_success = self._loader_result
            
            if not (vehicle_success and loader_success):
                self.get_logger().error(
                    "One of the actions failed. Stopping sequence."
                )
                self.get_logger().error(
                    f"vehicle_control result = [{vehicle_success}] | arm_control result = [{loader_success}]"
                )
                self.is_cancled = True
                return

            self.get_logger().info(
                "Both actions completed successfully."
            )

            time.sleep(1.0)

        if not self.is_cancled:
            self.get_logger().info("All targets completed.")

        self.state_reset()

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

                    rclpy.spin_once(self, timeout_sec=0.1)
    
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
        self.is_arm_reached = False
        self.node.pub_arm_target_pos.publish(msg)

    def set_bucket_target_pos(self, pos:float):
        msg = Float64()
        msg.data = pos
        self.is_bucket_reached = False
        self.node.pub_bucket_target_pos.publish(msg)

    def set_arm_move(self):
        pulse_bool(self.node.pub_arm_move)

    def set_bucket_move(self):
        pulse_bool(self.node.pub_bucket_move)

    def wait_until_reached(self, is_reached:callable, timeout:float=30.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            
            time.sleep(0.1)

            if is_reached():
                return True
            
        return False
    
    def send_loader_goal(self, j1:float=None, j2:float=None):
        self.set_enable_arm_ctrl()
        self.set_enable_arm_arm_ctrl()
        self.set_enable_arm_bucket_ctrl()

        if j1 is not None:
            self.set_arm_target_pos(j1)
            self.set_arm_move()
        if j2 is not None:
            self.set_bucket_target_pos(j2)
            self.set_bucket_move()

        j1_ret = True
        if j1 is not None:
            j1_ret = self.wait_until_reached(lambda : self.is_arm_reached)

        j2_ret = True
        if j2 is not None:
            j2_ret = self.wait_until_reached(lambda : self.is_bucket_reached)
            
        return j1_ret and j2_ret

    def arm_reached_callback(self, msg:Bool):
        if msg.data:
            self.is_arm_reached = True
            self.node.get_logger().info("Arm reached target position")
        else:
            self.is_arm_reached = False

    def bucket_reached_callback(self, msg:Bool):
        if msg.data:
            self.is_bucket_reached = True
            self.node.get_logger().info("Bucket reached target position")
        else:
            self.is_bucket_reached = False

class VehicleMotionController:
    def __init__(self, node:SequenceExecutor, kp:float, ki:float, kd:float, sat_cmd:float, max_cmd:float, min_cmd:float, max_i:float, min_i:float, tol:float=0.1):
        self.node = node

        self._vehicle_spd  : float = 0.0
        self._vehicle_dist : float = 0.0

        self.reset()
        self.set_control_gain(kp, ki, kd)
        self.set_control_limit(sat_cmd, max_cmd, min_cmd, max_i, min_i)
        self.tol = tol

    def set_control_gain(self, kp:float, ki:float, kd:float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_control_limit(self, sat_cmd:float, max_cmd:float, min_cmd:float, max_i:float, min_i:float):
        self.sat_cmd = sat_cmd
        self.max_cmd = max_cmd
        self.min_cmd = min_cmd
        self.max_i   = max_i
        self.min_i   = min_i

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
    
    def control_pos(self, target_pos:float, timeout:float=5.0):

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
        # v_cmd         = np.clip(v_cmd, self.min_cmd, self.max_cmd)

        v_cmd = self.sat_cmd if self.err > 0 else -self.sat_cmd
        v_cmd = np.clip(v_cmd, self.min_cmd, self.max_cmd)
        return v_cmd, self.err

    def set_linear_spd(self, spd:float):
        msg_spd = Float64()
        msg_spd.data = spd
        self.node.pub_linear_spd.publish(msg_spd)
        pulse_bool(self.node.pub_linear_mov)

    def set_linear_stop_mov(self):
        pulse_bool(self.node.pub_linear_stop)

def main(args=None):
    rclpy.init(args=args)

    node = SequenceExecutor()
    node.get_logger().info("Wheel loader action server for load/unload start!!")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()