import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import csv
import time


class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')
        self.subscription = self.create_subscription(
            JointState,
            '/world/spring_mass_world/model/spring_mass_damper/joint_state',
            self.callback,
            10
        )
        self.data_log = []
        self.start_time = time.time()
        self.get_logger().info('Sensor node started. Listening for joint states...')

    def callback(self, msg):
        t = time.time() - self.start_time

        for i, name in enumerate(msg.name):
            if name == 'spring_joint':
                pos = msg.position[i] if i < len(msg.position) else 0.0
                vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
                eff = msg.effort[i] if i < len(msg.effort) else 0.0

                self.data_log.append([t, pos, vel, eff])
                self.get_logger().info(
                    f't={t:.3f}  pos={pos:.4f}  vel={vel:.4f}  eff={eff:.4f}'
                )
                break

    def save_data(self):
        filename = 'vibration_data.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'position', 'velocity', 'effort'])
            writer.writerows(self.data_log)
        self.get_logger().info(f'Saved {len(self.data_log)} samples to {filename}')


def main(args=None):
    rclpy.init(args=args)
    node = SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
