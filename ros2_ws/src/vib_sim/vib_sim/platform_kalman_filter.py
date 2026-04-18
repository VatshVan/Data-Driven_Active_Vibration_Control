import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import time


class PlatformKalmanFilter(Node):
    def __init__(self):
        super().__init__("platform_kalman_filter")

        self.joint_names = ["spring_fl", "spring_fr", "spring_rl", "spring_rr"]
        self.n_joints = 4

        # State: [pos_fl, vel_fl, pos_fr, vel_fr, pos_rl, vel_rl, pos_rr, vel_rr]
        self.n_states = 8

        # Initialize state estimate
        self.x = np.zeros(self.n_states)
        self.x[0] = -1.0  # fl equilibrium
        self.x[2] = -1.0  # fr equilibrium
        self.x[4] = -1.0  # rl equilibrium
        self.x[6] = -1.0  # rr equilibrium

        # State covariance
        self.P = np.eye(self.n_states) * 0.1

        # Process noise (how much we trust the model)
        self.Q = np.eye(self.n_states) * 0.01
        # Higher noise for velocities (less predictable)
        for i in range(4):
            self.Q[2*i+1, 2*i+1] = 0.05

        # Measurement noise (how much we trust sensors)
        # We measure positions and velocities
        self.R = np.eye(self.n_states) * 0.001
        # Velocity measurements are noisier
        for i in range(4):
            self.R[2*i+1, 2*i+1] = 0.01

        # Measurement matrix (we observe all states)
        self.H = np.eye(self.n_states)

        # System parameters
        self.k = 200.0   # spring stiffness
        self.c = 10.0    # damping
        self.m = 5.0     # mass
        self.eq_pos = -1.0

        self.dt = 0.01
        self.prev_time = None
        self.initialized = False

        # Subscribe to raw joint states
        self.sub = self.create_subscription(
            JointState,
            "/world/platform_world/model/vibration_platform/joint_state",
            self.callback,
            10
        )

        # Publish filtered joint states
        self.pub = self.create_publisher(
            JointState,
            "/platform/filtered_joint_state",
            10
        )

        self.count = 0
        self.get_logger().info("Platform Kalman Filter started!")
        self.get_logger().info(f"  Spring: {self.k} N/m, Damping: {self.c} Ns/m, Mass: {self.m} kg")

    def get_F_matrix(self, dt):
        """State transition matrix for 4 independent spring-mass-damper systems"""
        F = np.eye(self.n_states)
        for i in range(4):
            pi = 2 * i      # position index
            vi = 2 * i + 1  # velocity index
            # pos_new = pos + vel * dt
            F[pi, vi] = dt
            # vel_new = vel + (-k/m * (pos - eq) - c/m * vel) * dt
            F[vi, pi] = -self.k / self.m * dt
            F[vi, vi] = 1.0 - self.c / self.m * dt
        return F

    def predict(self, dt):
        """Prediction step"""
        F = self.get_F_matrix(dt)

        # State prediction
        self.x = F @ self.x
        # Add spring force toward equilibrium in prediction
        for i in range(4):
            pi = 2 * i
            # The F matrix already handles spring dynamics

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """Measurement update step"""
        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P

    def callback(self, msg):
        current_time = time.time()

        if self.prev_time is None:
            self.prev_time = current_time
            # Initialize state from first measurement
            for target_name in self.joint_names:
                for i, name in enumerate(msg.name):
                    if name == target_name:
                        idx = self.joint_names.index(target_name)
                        self.x[2*idx] = msg.position[i] if i < len(msg.position) else -1.0
                        self.x[2*idx+1] = msg.velocity[i] if i < len(msg.velocity) else 0.0
                        break
            self.initialized = True
            return

        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 0.01
        self.prev_time = current_time

        # Build measurement vector
        z = np.zeros(self.n_states)
        for target_name in self.joint_names:
            for i, name in enumerate(msg.name):
                if name == target_name:
                    idx = self.joint_names.index(target_name)
                    z[2*idx] = msg.position[i] if i < len(msg.position) else 0.0
                    z[2*idx+1] = msg.velocity[i] if i < len(msg.velocity) else 0.0
                    break

        # Kalman filter steps
        self.predict(dt)
        self.update(z)

        # Publish filtered state
        filtered_msg = JointState()
        filtered_msg.header.stamp = self.get_clock().now().to_msg()
        filtered_msg.name = list(self.joint_names)
        filtered_msg.position = [float(self.x[2*i]) for i in range(4)]
        filtered_msg.velocity = [float(self.x[2*i+1]) for i in range(4)]
        self.pub.publish(filtered_msg)

        self.count += 1
        if self.count % 200 == 0:
            raw_str = "  ".join([f"{self.joint_names[i]}: raw={z[2*i]:.4f} filt={self.x[2*i]:.4f}" for i in range(4)])
            self.get_logger().info(f"KF #{self.count}  {raw_str}")


def main(args=None):
    rclpy.init(args=args)
    node = PlatformKalmanFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
