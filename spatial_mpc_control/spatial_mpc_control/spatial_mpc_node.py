import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import torch
import numpy as np
import scipy.sparse as sp

from spatial_mpc_control.model import SpatialMPCOrchestrator

class SpatialMPCNode(Node):
    def __init__(self):
        super().__init__('spatial_mpc_node')

        self.declare_parameter('dt', 0.01)
        self.declare_parameter('mpc_horizon', 20)
        self.declare_parameter('mass_nominal', 12.0)

        dt = self.get_parameter('dt').value
        N = self.get_parameter('mpc_horizon').value
        m_nom = self.get_parameter('mass_nominal').value

        pos_arr = [(0.5, 0.4), (0.5, -0.4), (-0.5, 0.4), (-0.5, -0.4)]
        k_arr = [150.0, 150.0, 150.0, 150.0]
        c_arr = [20.0, 20.0, 20.0, 20.0]

        Q = sp.csc_matrix(np.diag([100.0, 80.0, 80.0, 5.0, 5.0, 5.0]))
        R = sp.csc_matrix(np.diag([0.5, 0.5, 0.5, 0.5]))
        u_min = [-50.0, -50.0, -50.0, -50.0]
        u_max = [50.0, 50.0, 50.0, 50.0]

        self.orchestrator = SpatialMPCOrchestrator(
            m_init=m_nom, Ixx=2.5, Iyy=3.2,
            k_arr=k_arr, c_arr=c_arr, pos_arr=pos_arr,
            N=N, Q=Q, R=R, u_min=u_min, u_max=u_max, dt=dt
        )

        self.z = 0.0
        self.phi = 0.0
        self.theta = 0.0
        self.z_dot = 0.0
        self.phi_dot = 0.0
        self.theta_dot = 0.0

        self.sub_imu = self.create_subscription(
            Imu, 
            '/platform/imu', 
            self.imu_callback, 
            10
        )
        
        self.sub_odom = self.create_subscription(
            Odometry, 
            '/platform/odom', 
            self.odom_callback, 
            10
        )

        self.pub_f1 = self.create_publisher(Wrench, '/actuator_fl_controller/command', 10)
        self.pub_f2 = self.create_publisher(Wrench, '/actuator_fr_controller/command', 10)
        self.pub_f3 = self.create_publisher(Wrench, '/actuator_rl_controller/command', 10)
        self.pub_f4 = self.create_publisher(Wrench, '/actuator_rr_controller/command', 10)

        self.timer = self.create_timer(dt, self.control_loop)

    def imu_callback(self, msg):
        q = msg.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.phi = np.arctan2(siny_cosp, cosy_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        self.theta = np.arcsin(np.clip(sinp, -1.0, 1.0))

        self.phi_dot = msg.angular_velocity.x
        self.theta_dot = msg.angular_velocity.y

    def odom_callback(self, msg):
        self.z = msg.pose.pose.position.z
        self.z_dot = msg.twist.twist.linear.z

    def _create_wrench(self, force_z):
        w = Wrench()
        w.force.z = float(force_z)
        return w

    def control_loop(self):
        z_meas = torch.tensor(
            [self.z, self.phi, self.theta, self.z_dot, self.phi_dot, self.theta_dot], 
            dtype=torch.float32
        )

        u_opt = self.orchestrator.dispatch_control(z_meas)

        self.pub_f1.publish(self._create_wrench(u_opt[0]))
        self.pub_f2.publish(self._create_wrench(u_opt[1]))
        self.pub_f3.publish(self._create_wrench(u_opt[2]))
        self.pub_f4.publish(self._create_wrench(u_opt[3]))

def main(args=None):
    rclpy.init(args=args)
    node = SpatialMPCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
