#!/usr/bin/env python3
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle

from wheel_loader_navigation.action import ForwardBucket
from yoko_loader_tester.action import Trigger
from std_msgs.msg import Bool, Int32, Int16, Float64

class ArmPLCController:
    def __init__(self, node:Node):
        self.node = node

        self.pub_enable_arm_ctrl = self.node.create_publisher(Bool, '/arm/enable_ctrl', 10)
        self.pub_enable_arm_arm_ctrl = self.node.create_publisher(Bool, '/arm/enable_arm_ctrl', 10)
        self.pub_enable_arm_bucket_ctrl = self.node.create_publisher(Bool, '/arm/enable_bucket_ctrl', 10)
        self.pub_arm_target_pos = self.node.create_publisher(Float64, '/arm/set_arm_target_pos', 10)
        self.pub_bucket_target_pos = self.node.create_publisher(Float64, '/arm/set_bucket_target_pos', 10)
        self.pub_arm_move = self.node.create_publisher(Bool, '/arm/arm_move_to_pos', 10)
        self.pub_bucket_move = self.node.create_publisher(Bool, '/arm/bucket_move_to_pos', 10)

        node.create_subscription(Bool, "/arm/is_arm_reached_target", self.arm_reached_callback, 10)
        node.create_subscription(Bool, "/arm/is_bucket_reached_target", self.bucket_reached_callback, 10)

        self.is_arm_reached = False
        self.is_bucket_reached = False

    def pulse_bool(self, publisher):
        msg = Bool()
        msg.data = True
        publisher.publish(msg)

        time.sleep(0.1)

        msg.data = False
        publisher.publish(msg)

    def set_enable_arm_ctrl(self):
        msg = Bool()
        msg.data = True
        self.pub_enable_arm_ctrl.publish(msg)

    def set_disable_arm_ctrl(self):
        msg = Bool()
        msg.data = False
        self.pub_enable_arm_ctrl.publish(msg)

    def set_enable_arm_arm_ctrl(self):
        msg = Bool()
        msg.data = True
        self.pub_enable_arm_arm_ctrl.publish(msg)

    def set_disable_arm_arm_ctrl(self):
        msg = Bool()
        msg.data = False
        self.pub_enable_arm_arm_ctrl.publish(msg)

    def set_enable_arm_bucket_ctrl(self):
        msg = Bool()
        msg.data = True
        self.pub_enable_arm_bucket_ctrl.publish(msg)

    def set_disable_arm_bucket_ctrl(self):
        msg = Bool()
        msg.data = False
        self.pub_enable_arm_bucket_ctrl.publish(msg)

    def set_arm_target_pos(self, pos:float):
        msg = Float64()
        msg.data = pos
        self.is_arm_reached = False
        self.pub_arm_target_pos.publish(msg)

    def set_bucket_target_pos(self, pos:float):
        msg = Float64()
        msg.data = pos
        self.is_bucket_reached = False
        self.pub_bucket_target_pos.publish(msg)

    def set_arm_move(self):
        self.pulse_bool(self.pub_arm_move)

    def set_bucket_move(self):
        self.pulse_bool(self.pub_bucket_move)

    def wait_until_reached(self, is_reached:bool, timeout:float=5.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            
            time.sleep(0.1)

            if is_reached:
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

        if j1 is not None:
            if self.wait_until_reached(self.is_arm_reached):
                return True

        if j2 is not None:
            if self.wait_until_reached(self.is_bucket_reached):
                return True
            
        return False

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

class SequenceExecutor(Node):

    def __init__(self):
        super().__init__("sequence_executor")

        self.armController = ArmPLCController(self)

        self.action_server = ActionServer(
            node             = self,
            action_type      = Trigger,
            action_name      = "/arm_loader/execute_sequence",
            execute_callback = self.execute_loader_goal
        )

        self.vehicle_client = ActionClient(
            self,
            ForwardBucket,
            "/forward_bucket"
        )

        self.targets = [
            [0.24197153387468573, 0.0, 0.0],
            [1.0, 0.21979246164780855, 0.3033429986776818],
            [2.0, 0.22009244819553492, 0.3037867161184603],
            [2.0, 0.22009244819553492, 0.3037867161184603],
            [0.0, 0.8029340032443635, 0.15712545046230594]
        ]

        self.action_mutex = threading.Lock()


    def send_vehicle_goal(self, x):

        self.vehicle_client.wait_for_server()

        goal = ForwardBucket.Goal()
        goal.distance = x

        send_goal_future = self.vehicle_client.send_goal_async(goal)

        self.action_mutex.acquire()
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.action_mutex.release()

        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Vehicle goal rejected")
            return False

        self.get_logger().info(f"Vehicle goal accepted: x={x}")

        result_future = goal_handle.get_result_async()

        self.action_mutex.acquire()
        rclpy.spin_until_future_complete(self, result_future)
        self.action_mutex.release()

        result = result_future.result()

        if result.status == 4:
            self.get_logger().info("Vehicle action succeeded")
            return True

        self.get_logger().error("Vehicle action failed")
        return False

    def execute_sequence(self):

        for idx, (x, j1, j2) in enumerate(self.targets):

            self.get_logger().info(
                f"########### Execute Target {idx+1}/{len(self.targets)} ####################\n"
            )

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

            vehicle_success = True
            if not (vehicle_success and loader_success):
                self.get_logger().error(
                    "One of the actions failed. Stopping sequence."
                )
                return

            self.get_logger().info(
                "Both actions completed successfully."
            )

            time.sleep(1.0)

        self.get_logger().info("All targets completed.")

    def execute_loader_goal(self, goal_handle: ServerGoalHandle):
        self.get_logger().info("Received request to execute loader sequence.")

        process_thread = threading.Thread(target=self.execute_sequence, daemon=True)
        process_thread.start()

        feedback = Trigger.Feedback()
        result = Trigger.Result()

        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info("Loader action canceled.")
                    self.sequence_cancel_callback()
                    goal_handle.canceled()
                    result.success = False
                    return result

                is_done, state_num = self.get_sequence_state()
                feedback.state_num = state_num  
                goal_handle.publish_feedback(feedback)

                if is_done:
                    self.get_logger().info("Loader sequence completed successfully.")
                    break

                rclpy.spin_once(self)
                time.sleep(0.05)

        except Exception as e:
            self.get_logger().error(f"Error during sequence execution: {e}")
            goal_handle.abort()
            result.success = False
            return result

        goal_handle.succeed()
        result.success = True
        return result

    def sequence_cancel_callback(self):
        raise NotImplementedError("Sequence cancel callback not implemented.")

    def get_sequence_state(self):
        raise NotImplementedError("Get sequence state method not implemented.")
    
def main(args=None):
    rclpy.init(args=args)

    node = SequenceExecutor()
    node.execute_sequence()
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()