import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
import numpy as np
import os
import torch
import torch.nn as nn


class ControlPolicy(nn.Module):
    """Same architecture as training script"""
    def __init__(self, state_dim=2, action_dim=1, max_force=100.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.max_force = max_force

    def forward(self, state):
        return self.net(state) * self.max_force


class NeuralControllerNode(Node):
    def __init__(self):
        super().__init__('neural_controller_node')

        # Load trained policy
        model_path = os.path.expanduser('~/ros2_ws/control_policy.pth')
        if not os.path.exists(model_path):
            self.get_logger().error(
                f'Model not found at {model_path}. Train first!'
            )
            return

        self.policy = ControlPolicy(state_dim=2, action_dim=1, max_force=100.0)
        self.policy.load_state_dict(torch.load(model_path, weights_only=True))
        self.policy.eval()
        self.get_logger().info('Neural MPC policy loaded successfully!')

        # Subscribe to estimated state
        self.state_sub = self.create_subscription(
            Float64MultiArray,
            '/estimated_state',
            self.control_callback,
            10
        )

        # Publish control force
        self.force_pub = self.create_publisher(
            Float64,
            '/model/spring_mass_damper/joint/spring_joint/cmd_force',
            10
        )

        self.get_logger().info('Neural controller ready!')

    def control_callback(self, msg):
        state = torch.tensor(msg.data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = self.policy(state).squeeze().item()

        # Publish force command
        force_msg = Float64()
        force_msg.data = float(action)
        self.force_pub.publish(force_msg)

        self.get_logger().info(
            f'State: [{msg.data[0]:.4f}, {msg.data[1]:.4f}] '
            f'-> Force: {action:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = NeuralControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
