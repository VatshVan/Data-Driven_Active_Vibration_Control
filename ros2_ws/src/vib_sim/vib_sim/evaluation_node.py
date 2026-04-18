import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import torch
import torch.nn as nn
import numpy as np
import math
import time
import os
import csv


class PlatformControlPolicy(nn.Module):
    def __init__(self, max_force=80.0):
        super(PlatformControlPolicy, self).__init__()
        self.max_force = max_force
        self.net = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state) * self.max_force


class EvaluationNode(Node):
    def __init__(self):
        super().__init__("evaluation_node")

        self.declare_parameter("mode", "uncontrolled")
        self.mode = self.get_parameter("mode").get_parameter_value().string_value

        self.declare_parameter("duration", 30.0)
        self.duration = self.get_parameter("duration").get_parameter_value().double_value

        self.joint_names = ["spring_fl", "spring_fr", "spring_rl", "spring_rr"]
        self.eq_pos = -1.0

        self.sub = self.create_subscription(
            JointState,
            "/world/platform_world/model/vibration_platform/joint_state",
            self.joint_callback,
            10
        )

        self.force_pubs = {}
        for name in self.joint_names:
            topic = f"/model/vibration_platform/joint/{name}/cmd_force"
            self.force_pubs[name] = self.create_publisher(Float64, topic, 10)

        # Neural MPC
        self.policy = None
        if self.mode == "controlled":
            model_path = os.path.expanduser("~/ros2_ws/platform_control_policy.pth")
            self.policy = PlatformControlPolicy(max_force=80.0)
            self.policy.load_state_dict(torch.load(model_path, weights_only=True))
            self.policy.eval()
            self.get_logger().info(f"Loaded neural controller from {model_path}")

        # PID
        self.Kp = 150.0
        self.Ki = 20.0
        self.Kd = 30.0
        self.max_force = 80.0
        self.integral = {name: 0.0 for name in self.joint_names}
        self.prev_error = {name: 0.0 for name in self.joint_names}
        self.prev_step_time = None

        self.positions = {name: 0.0 for name in self.joint_names}
        self.velocities = {name: 0.0 for name in self.joint_names}
        self.data_received = False

        self.data_log = []
        self.start_time = time.time()

        self.timer = self.create_timer(0.01, self.step)
        self.t = 0.0
        self.dt = 0.01

        self.get_logger().info(f"Evaluation started: mode={self.mode}, duration={self.duration}s")

    def joint_callback(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                self.positions[name] = msg.position[i] if i < len(msg.position) else 0.0
                self.velocities[name] = msg.velocity[i] if i < len(msg.velocity) else 0.0
        self.data_received = True

    def compute_pid(self):
        current_time = time.time()
        if self.prev_step_time is None:
            self.prev_step_time = current_time
            return {name: 0.0 for name in self.joint_names}

        dt = current_time - self.prev_step_time
        if dt <= 0:
            dt = 0.01
        self.prev_step_time = current_time

        forces = {}
        for name in self.joint_names:
            error = self.eq_pos - self.positions[name]
            self.integral[name] += error * dt
            if self.integral[name] > 5.0:
                self.integral[name] = 5.0
            elif self.integral[name] < -5.0:
                self.integral[name] = -5.0
            derivative = (error - self.prev_error[name]) / dt
            force = self.Kp * error + self.Ki * self.integral[name] + self.Kd * derivative
            if force > self.max_force:
                force = self.max_force
            elif force < -self.max_force:
                force = -self.max_force
            forces[name] = force
            self.prev_error[name] = error
        return forces

    def step(self):
        if not self.data_received:
            return

        elapsed = time.time() - self.start_time
        if elapsed > self.duration:
            self.save_data()
            self.timer.cancel()
            self.get_logger().info("Evaluation complete!")
            return

        disturbance = {
            "spring_fl": 30.0 * math.sin(2.0 * math.pi * 2.0 * self.t) +
                         15.0 * math.sin(2.0 * math.pi * 7.0 * self.t),
            "spring_fr": 25.0 * math.sin(2.0 * math.pi * 3.0 * self.t) +
                         10.0 * math.sin(2.0 * math.pi * 8.0 * self.t),
            "spring_rl": 20.0 * math.sin(2.0 * math.pi * 4.0 * self.t) +
                         12.0 * math.sin(2.0 * math.pi * 9.0 * self.t),
            "spring_rr": 35.0 * math.sin(2.0 * math.pi * 5.0 * self.t) +
                         8.0 * math.sin(2.0 * math.pi * 6.0 * self.t),
        }

        control_forces = {name: 0.0 for name in self.joint_names}

        if self.mode == "controlled" and self.policy is not None:
            state = []
            for name in self.joint_names:
                state.append(self.positions[name])
                state.append(self.velocities[name])
            state_tensor = torch.FloatTensor([state])
            with torch.no_grad():
                forces_out = self.policy(state_tensor)
            forces_np = forces_out.numpy()[0]
            for j, name in enumerate(self.joint_names):
                control_forces[name] = float(forces_np[j])

        elif self.mode == "pid":
            control_forces = self.compute_pid()

        for name in self.joint_names:
            msg = Float64()
            msg.data = disturbance[name] + control_forces[name]
            self.force_pubs[name].publish(msg)

        row = [elapsed, self.t]
        for name in self.joint_names:
            row.append(self.positions[name])
            row.append(self.velocities[name])
        for name in self.joint_names:
            row.append(disturbance[name])
        for name in self.joint_names:
            row.append(control_forces[name])
        self.data_log.append(row)

        self.t += self.dt

        if len(self.data_log) % 500 == 0:
            self.get_logger().info(f"Recording... {elapsed:.1f}s / {self.duration}s")

    def save_data(self):
        filename = os.path.expanduser(f"~/ros2_ws/eval_{self.mode}.csv")
        header = ["elapsed", "sim_time"]
        for name in self.joint_names:
            header.extend([f"pos_{name}", f"vel_{name}"])
        for name in self.joint_names:
            header.append(f"dist_{name}")
        for name in self.joint_names:
            header.append(f"ctrl_{name}")

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.data_log)

        self.get_logger().info(f"Saved {len(self.data_log)} samples to {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
