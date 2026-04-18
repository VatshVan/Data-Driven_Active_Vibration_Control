import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
import time
import os
import math


class PlatformDataCollector(Node):
    def __init__(self):
        super().__init__('platform_data_collector')

        # Joint names
        self.joint_names = ['spring_fl', 'spring_fr', 'spring_rl', 'spring_rr']

        # Subscribe to joint states (all 4 joints come in one message)
        self.joint_sub = self.create_subscription(
            JointState,
            '/world/platform_world/model/vibration_platform/joint_state',
            self.joint_callback,
            10
        )

        # Publishers: one force command per actuator
        self.force_pubs = {}
        for name in self.joint_names:
            topic = f'/model/vibration_platform/joint/{name}/cmd_force'
            self.force_pubs[name] = self.create_publisher(Float64, topic, 10)

        # Data storage
        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.current_velocities = {name: 0.0 for name in self.joint_names}
        self.data_received = False
        self.dataset = []
        self.start_time = time.time()

        # Timer: collect at 100 Hz
        self.timer = self.create_timer(0.01, self.collect_step)

        # Collection parameters
        self.max_samples = 15000
        self.collected = 0

        self.get_logger().info(
            f'Platform data collector started. Target: {self.max_samples} samples'
        )
        self.get_logger().info(f'Actuators: {self.joint_names}')

    def joint_callback(self, msg):
        """Parse joint states for all 4 actuators"""
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                self.current_positions[name] = msg.position[i] if i < len(msg.position) else 0.0
                self.current_velocities[name] = msg.velocity[i] if i < len(msg.velocity) else 0.0
        self.data_received = True

    def collect_step(self):
        """Apply random forces and record data"""
        if not self.data_received:
            return

        if self.collected >= self.max_samples:
            self.save_dataset()
            self.timer.cancel()
            return

        t = time.time() - self.start_time

        # Generate random + sinusoidal forces for each actuator
        forces = {}
        for i, name in enumerate(self.joint_names):
            # Random force
            random_f = np.random.uniform(-80.0, 80.0)

            # Different frequency sinusoidal for each actuator
            freq = 2.0 + i * 1.5  # FL=2Hz, FR=3.5Hz, RL=5Hz, RR=6.5Hz
            sin_f = 40.0 * math.sin(2.0 * math.pi * freq * t)

            # Occasional impulse
            impulse = 0.0
            if np.random.random() < 0.01:  # 1% chance each step
                impulse = np.random.uniform(-200.0, 200.0)

            forces[name] = random_f + sin_f + impulse

        # Publish forces
        for name in self.joint_names:
            msg = Float64()
            msg.data = float(forces[name])
            self.force_pubs[name].publish(msg)

        # Record data
        # Format: [time, pos_fl, vel_fl, pos_fr, vel_fr, pos_rl, vel_rl, pos_rr, vel_rr,
        #          force_fl, force_fr, force_rl, force_rr]
        sample = [t]
        for name in self.joint_names:
            sample.append(self.current_positions[name])
            sample.append(self.current_velocities[name])
        for name in self.joint_names:
            sample.append(forces[name])

        self.dataset.append(sample)
        self.collected += 1

        if self.collected % 2000 == 0:
            self.get_logger().info(
                f'Collected {self.collected}/{self.max_samples} samples'
            )
            # Print current state for monitoring
            pos_str = '  '.join(
                [f'{n}: pos={self.current_positions[n]:.3f} vel={self.current_velocities[n]:.3f}'
                 for n in self.joint_names]
            )
            self.get_logger().info(f'  {pos_str}')

    def save_dataset(self):
        """Save collected data to files"""
        data = np.array(self.dataset)

        # Column names
        header_parts = ['time']
        for name in self.joint_names:
            header_parts.append(f'pos_{name}')
            header_parts.append(f'vel_{name}')
        for name in self.joint_names:
            header_parts.append(f'force_{name}')
        header = ','.join(header_parts)

        # Save as NPY
        npy_path = os.path.expanduser('~/ros2_ws/platform_training_data.npy')
        np.save(npy_path, data)

        # Save as CSV
        csv_path = os.path.expanduser('~/ros2_ws/platform_training_data.csv')
        np.savetxt(csv_path, data, delimiter=',', header=header, comments='')

        self.get_logger().info(f'=== Dataset Saved! ===')
        self.get_logger().info(f'  Samples: {len(self.dataset)}')
        self.get_logger().info(f'  Columns: {len(header_parts)}')
        self.get_logger().info(f'  Header: {header}')
        self.get_logger().info(f'  NPY: {npy_path}')
        self.get_logger().info(f'  CSV: {csv_path}')


def main(args=None):
    rclpy.init(args=args)
    node = PlatformDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_dataset()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '_main_':
    main()
