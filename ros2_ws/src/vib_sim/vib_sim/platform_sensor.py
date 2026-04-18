import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import csv
import time


class PlatformSensor(Node):
    def __init__(self):
        super().__init__('platform_sensor')

        self.joint_names = ['spring_fl', 'spring_fr', 'spring_rl', 'spring_rr']

        self.sub = self.create_subscription(
            JointState,
            '/world/platform_world/model/vibration_platform/joint_state',
            self.callback,
            10
        )

        self.data_log = []
        self.start_time = time.time()
        self.get_logger().info('Platform sensor started. Monitoring 4 actuators...')

    def callback(self, msg):
        t = time.time() - self.start_time

        row = [t]
        log_parts = []

        for target_name in self.joint_names:
            found = False
            for i, name in enumerate(msg.name):
                if name == target_name:
                    pos = msg.position[i] if i < len(msg.position) else 0.0
                    vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
                    row.extend([pos, vel])
                    log_parts.append(f'{target_name}: p={pos:.3f} v={vel:.3f}')
                    found = True
                    break
            if not found:
                row.extend([0.0, 0.0])

        self.data_log.append(row)

        if len(self.data_log) % 100 == 0:
            self.get_logger().info(f't={t:.2f}  ' + '  '.join(log_parts))

    def save_data(self):
        filename = 'platform_sensor_data.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['time']
            for name in self.joint_names:
                header.extend([f'pos_{name}', f'vel_{name}'])
            writer.writerow(header)
            writer.writerows(self.data_log)
        self.get_logger().info(f'Saved {len(self.data_log)} samples to {filename}')


def main(args=None):
    rclpy.init(args=args)
    node = PlatformSensor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '_main_':
    main()
