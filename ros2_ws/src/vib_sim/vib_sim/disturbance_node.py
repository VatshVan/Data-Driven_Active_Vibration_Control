import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import math


class DisturbanceNode(Node):
    def __init__(self):
        super().__init__('disturbance_node')
        self.publisher = self.create_publisher(
            Float64,
            '/model/spring_mass_damper/joint/spring_joint/cmd_force',
            10
        )
        self.timer = self.create_timer(0.01, self.publish_force)
        self.t = 0.0
        self.dt = 0.01
        self.get_logger().info('Disturbance node started. Applying forces...')

    def publish_force(self):
        force = (
            20.0 * math.sin(2.0 * math.pi * 2.0 * self.t) +
            10.0 * math.sin(2.0 * math.pi * 5.0 * self.t) +
            5.0 * math.sin(2.0 * math.pi * 10.0 * self.t)
        )

        msg = Float64()
        msg.data = force
        self.publisher.publish(msg)
        self.t += self.dt


def main(args=None):
    rclpy.init(args=args)
    node = DisturbanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
