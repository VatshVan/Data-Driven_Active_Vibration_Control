import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import time


class PlatformPIDController(Node):
    def __init__(self):
        super().__init__("platform_pid_controller")

        self.joint_names = ["spring_fl", "spring_fr", "spring_rl", "spring_rr"]
        self.eq_pos = -1.0

        # PID gains (tuned for spring 200 N/m, damping 10 Ns/m, mass 5kg)
        self.Kp = 150.0
        self.Ki = 20.0
        self.Kd = 30.0
        self.max_force = 80.0

        self.integral = {name: 0.0 for name in self.joint_names}
        self.prev_error = {name: 0.0 for name in self.joint_names}
        self.prev_time = None

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
        self.get_logger().info(f"PID Controller started: Kp={self.Kp} Ki={self.Ki} Kd={self.Kd}")

    def callback(self, msg):
        current_time = time.time()
        if self.prev_time is None:
            self.prev_time = current_time
            return

        dt = current_time - self.prev_time
        if dt <= 0:
            return
        self.prev_time = current_time

        positions = {}
        velocities = {}
        for target_name in self.joint_names:
            for i, name in enumerate(msg.name):
                if name == target_name:
                    positions[target_name] = msg.position[i] if i < len(msg.position) else 0.0
                    velocities[target_name] = msg.velocity[i] if i < len(msg.velocity) else 0.0
                    break

        forces = {}
        for name in self.joint_names:
            if name not in positions:
                continue

            error = self.eq_pos - positions[name]

            self.integral[name] += error * dt
            # Anti-windup: clamp integral
            if self.integral[name] > 5.0:
                self.integral[name] = 5.0
            elif self.integral[name] < -5.0:
                self.integral[name] = -5.0

            derivative = (error - self.prev_error[name]) / dt

            force = self.Kp * error + self.Ki * self.integral[name] + self.Kd * derivative

            # Clamp force
            if force > self.max_force:
                force = self.max_force
            elif force < -self.max_force:
                force = -self.max_force

            forces[name] = force
            self.prev_error[name] = error

        for name in self.joint_names:
            if name in forces:
                msg_out = Float64()
                msg_out.data = forces[name]
                self.force_pubs[name].publish(msg_out)

        self.control_count += 1
        if self.control_count % 100 == 0:
            fl = forces.get("spring_fl", 0.0)
            fr = forces.get("spring_fr", 0.0)
            rl = forces.get("spring_rl", 0.0)
            rr = forces.get("spring_rr", 0.0)
            pfl = positions.get("spring_fl", 0.0)
            pfr = positions.get("spring_fr", 0.0)
            prl = positions.get("spring_rl", 0.0)
            prr = positions.get("spring_rr", 0.0)
            self.get_logger().info(
                f"PID #{self.control_count}  "
                f"FL:{fl:.1f} FR:{fr:.1f} RL:{rl:.1f} RR:{rr:.1f}  "
                f"Pos: {pfl:.3f} {pfr:.3f} {prl:.3f} {prr:.3f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PlatformPIDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
