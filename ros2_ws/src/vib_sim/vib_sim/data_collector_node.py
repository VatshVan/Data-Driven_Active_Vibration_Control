import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
import numpy as np
import time
import os


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')

        # Subscribe to estimated state from Kalman filter
        self.state_sub = self.create_subscription(
            Float64MultiArray,
            '/estimated_state',
            self.state_callback,
            10
        )

        # Subscribe to raw joint state for ground truth
        from sensor_msgs.msg import JointState
        self.joint_sub = self.create_subscription(
            JointState,
            '/world/spring_mass_world/model/spring_mass_damper/joint_state',
            self.joint_callback,
            10
        )

        # Publisher: send random exploration forces
        self.force_pub = self.create_publisher(
            Float64,
            '/model/spring_mass_damper/joint/spring_joint/cmd_force',
            10
        )

        # Data storage
        self.current_state = None
        self.current_raw_pos = None
        self.current_raw_vel = None
        self.dataset = []
        self.start_time = time.time()

        # Timer: collect data at 100 Hz
        self.timer = self.create_timer(0.01, self.collect_step)

        # Collection parameters
        self.max_samples = 10000
        self.collected = 0

        self.get_logger().info(
            f'Data collector started. Collecting {self.max_samples} samples...'
        )

    def state_callback(self, msg):
        self.current_state = np.array(msg.data)

    def joint_callback(self, msg):
        for i, name in enumerate(msg.name):
            if name == 'spring_joint':
                self.current_raw_pos = msg.position[i]
                self.current_raw_vel = msg.velocity[i]
                break

    def collect_step(self):
        if self.current_state is None:
            return

        if self.collected >= self.max_samples:
            self.save_dataset()
            self.timer.cancel()
            return

        # Generate random exploration force
        # Mix of random + sinusoidal for good coverage
        t = time.time() - self.start_time
        random_force = np.random.uniform(-100.0, 100.0)
        sin_force = 50.0 * np.sin(2.0 * np.pi * 3.0 * t)
        force = random_force + sin_force

        # Publish force
        msg = Float64()
        msg.data = float(force)
        self.force_pub.publish(msg)

        # Record: [time, est_pos, est_vel, raw_pos, raw_vel, force_applied]
        raw_pos = self.current_raw_pos if self.current_raw_pos is not None else 0.0
        raw_vel = self.current_raw_vel if self.current_raw_vel is not None else 0.0

        sample = [
            t,
            self.current_state[0],   # estimated position
            self.current_state[1],   # estimated velocity
            raw_pos,                  # raw position
            raw_vel,                  # raw velocity
            force                     # applied force
        ]
        self.dataset.append(sample)
        self.collected += 1

        if self.collected % 1000 == 0:
            self.get_logger().info(
                f'Collected {self.collected}/{self.max_samples} samples'
            )

    def save_dataset(self):
        data = np.array(self.dataset)
        
        # Save as numpy file
        save_path = os.path.expanduser('~/ros2_ws/training_data.npy')
        np.save(save_path, data)

        # Also save as CSV for easy viewing
        csv_path = os.path.expanduser('~/ros2_ws/training_data.csv')
        header = 'time,est_pos,est_vel,raw_pos,raw_vel,force'
        np.savetxt(csv_path, data, delimiter=',', header=header, comments='')

        self.get_logger().info(
            f'Dataset saved! {len(self.dataset)} samples'
        )
        self.get_logger().info(f'  NPY: {save_path}')
        self.get_logger().info(f'  CSV: {csv_path}')


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_dataset()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
