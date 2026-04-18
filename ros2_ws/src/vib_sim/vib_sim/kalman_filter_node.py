import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np


class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')

        # System parameters
        m = 5.0
        k = 200.0
        c = 10.0
        dt = 0.001

        # State transition matrix [position, velocity]
        self.A = np.array([
            [1.0, dt],
            [-k / m * dt, 1.0 - c / m * dt]
        ])

        # Measurement matrix (we measure position)
        self.H = np.array([[1.0, 0.0]])

        # Process noise covariance
        self.Q = np.diag([1e-4, 1e-3])

        # Measurement noise covariance
        self.R = np.array([[1e-2]])

        # Initial state estimate [position, velocity]
        self.x_hat = np.array([[0.0], [0.0]])

        # Initial error covariance
        self.P = np.eye(2) * 0.1

        # Subscriber: raw joint states from Gazebo
        self.sub = self.create_subscription(
            JointState,
            '/world/spring_mass_world/model/spring_mass_damper/joint_state',
            self.callback,
            10
        )

        # Publisher: estimated state
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/estimated_state',
            10
        )

        self.get_logger().info('Kalman filter node started.')

    def callback(self, msg):
        for i, name in enumerate(msg.name):
            if name == 'spring_joint':
                # Get measurement (position)
                z = np.array([[msg.position[i]]])

                # === PREDICT ===
                x_pred = self.A @ self.x_hat
                P_pred = self.A @ self.P @ self.A.T + self.Q

                # === UPDATE ===
                S = self.H @ P_pred @ self.H.T + self.R
                K = P_pred @ self.H.T @ np.linalg.inv(S)
                self.x_hat = x_pred + K @ (z - self.H @ x_pred)
                self.P = (np.eye(2) - K @ self.H) @ P_pred

                # Publish estimated state [position, velocity]
                state_msg = Float64MultiArray()
                state_msg.data = [
                    float(self.x_hat[0, 0]),
                    float(self.x_hat[1, 0])
                ]
                self.pub.publish(state_msg)

                self.get_logger().info(
                    f'Estimated pos={self.x_hat[0,0]:.4f}  '
                    f'vel={self.x_hat[1,0]:.4f}'
                )
                break


def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
