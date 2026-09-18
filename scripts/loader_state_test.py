#!/usr/bin/env python3
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from wheel_loader_navigation.action import ForwardBucket
from bucket_controller.action import SetTargetAngle

class SequenceExecutor(Node):

    def __init__(self):
        super().__init__("sequence_executor")

        self.vehicle_client = ActionClient(
            self,
            ForwardBucket,
            "/forward_bucket"
        )

        self.loader_client = ActionClient(
            self,
            SetTargetAngle,
            "/set_joint_target_angle"
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

    def send_loader_goal(self, j1, j2):

        self.loader_client.wait_for_server()
        goal = SetTargetAngle.Goal()

        goal.target_arm = j1
        goal.target_bucket = j2

        send_goal_future = self.loader_client.send_goal_async(goal)

        self.action_mutex.acquire()
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.action_mutex.release()

        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Loader goal rejected")
            return False

        self.get_logger().info(
            f"Loader goal accepted: j1={j1}, j2={j2}"
        )

        result_future = goal_handle.get_result_async()

        self.action_mutex.acquire()
        rclpy.spin_until_future_complete(self, result_future)
        self.action_mutex.release()
        
        result = result_future.result()

        if result.status == 4:
            self.get_logger().info("Loader action succeeded")
            return True

        self.get_logger().error("Loader action failed")
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
                    self.send_loader_goal(j1, j2)
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


def main(args=None):
    rclpy.init(args=args)

    node = SequenceExecutor()
    node.execute_sequence()
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()