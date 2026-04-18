import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import torch
import torch.nn as nn
import numpy as np
import os


class PlatformControlPolicy(nn.Module):
    def __init__(self, max_force=80.0):
        super(PlatformControlPolicy, self).__init__()
        self.max_force = max_force
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state) * self.max_force


class PlatformNeuralController(Node):
    def __init__(self):
        super().__init__("platform_neural_controller")

        self.joint_names = ["spring_fl", "spring_fr", "spring_rl", "spring_rr"]

        model_path = os.path.expanduser("~/ros2_ws/platform_control_policy.pth")
        self.policy = PlatformControlPolicy(max_force=80.0)
        self.policy.load_state_dict(torch.load(model_path, weights_only=True))
        self.policy.eval()
        self.get_logger().info(f"Loaded policy from {model_path}")

        self.sub = self.create_subscription(
            JointState,
            "/world/platform_world/model/vibration_platform/joint_state",
            self.callback,
            10
        )

        self.force_pubs = {}
        for name in self.joint_names:
            topic = f"/model/vibration_platform/joint/{name}/cmd_force"
            self.force_pubs[name] = self.create_publisher(Float64, topic, 10)

        self.control_count = 0
        self.get_logger().info("Platform neural controller started!")

    def callback(self, msg):
        positions = []
        velocities = []

        for target_name in self.joint_names:
            found = False
            for i, name in enumerate(msg.name):
                if name == target_name:
                    pos = msg.position[i] if i < len(msg.position) else 0.0
                    vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
                    positions.append(pos)
                    velocities.append(vel)
                    found = True
                    break
            if not found:
                positions.append(0.0)
                velocities.append(0.0)

        state = []
        for j in range(4):
            state.append(positions[j])
            state.append(velocities[j])

        state_tensor = torch.FloatTensor([state])

        with torch.no_grad():
            forces = self.policy(state_tensor)

        forces_np = forces.numpy()[0]

        for j, name in enumerate(self.joint_names):
            msg_out = Float64()
            msg_out.data = float(forces_np[j])
            self.force_pubs[name].publish(msg_out)

        self.control_count += 1
        if self.control_count % 100 == 0:
            self.get_logger().info(
                f"Control #{self.control_count}  "
                f"FL:{forces_np[0]:.1f} FR:{forces_np[1]:.1f} "
                f"RL:{forces_np[2]:.1f} RR:{forces_np[3]:.1f}  "
                f"Pos: {positions[0]:.3f} {positions[1]:.3f} {positions[2]:.3f} {positions[3]:.3f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PlatformNeuralController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
