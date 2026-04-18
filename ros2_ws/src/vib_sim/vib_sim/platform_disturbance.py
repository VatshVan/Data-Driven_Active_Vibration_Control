import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import math


class PlatformDisturbance(Node):
    def __init__(self):
        super().__init__("platform_disturbance")

        self.joint_names = ["spring_fl", "spring_fr", "spring_rl", "spring_rr"]

        self.force_pubs = {}
        for name in self.joint_names:
            topic = f"/model/vibration_platform/joint/{name}/cmd_force"
            self.force_pubs[name] = self.create_publisher(Float64, topic, 10)

        self.timer = self.create_timer(0.01, self.publish_forces)
        self.t = 0.0
        self.dt = 0.01

        self.get_logger().info("Platform disturbance node started (4 actuators)")

    def publish_forces(self):
        forces = {
            "spring_fl": 30.0 * math.sin(2.0 * math.pi * 2.0 * self.t) +
                         15.0 * math.sin(2.0 * math.pi * 7.0 * self.t),

            "spring_fr": 25.0 * math.sin(2.0 * math.pi * 3.0 * self.t) +
                         10.0 * math.sin(2.0 * math.pi * 8.0 * self.t),

            "spring_rl": 20.0 * math.sin(2.0 * math.pi * 4.0 * self.t) +
                         12.0 * math.sin(2.0 * math.pi * 9.0 * self.t),

            "spring_rr": 35.0 * math.sin(2.0 * math.pi * 5.0 * self.t) +
                         8.0 * math.sin(2.0 * math.pi * 6.0 * self.t),
        }

        for name in self.joint_names:
            msg = Float64()
            msg.data = forces[name]
            self.force_pubs[name].publish(msg)

        self.t += self.dt


def main(args=None):
    rclpy.init(args=args)
    node = PlatformDisturbance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
