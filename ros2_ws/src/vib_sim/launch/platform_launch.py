import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('vib_sim')
    sdf_file = os.path.join(pkg_dir, 'worlds', 'platform_world.sdf')

    return LaunchDescription([

        # Launch Gazebo
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', '-v', '4', sdf_file],
            output='screen',
            name='gazebo'
        ),

        # Start ROS-Gazebo bridge after a short delay
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',

                        # Joint states
                        '/world/platform_world/model/vibration_platform/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model]',

                        # Force commands for each actuator
                        '/model/vibration_platform/joint/spring_fl/cmd_force@std_msgs/msg/Float64[gz.msgs.Double]',
                        '/model/vibration_platform/joint/spring_fr/cmd_force@std_msgs/msg/Float64[gz.msgs.Double]',
                        '/model/vibration_platform/joint/spring_rl/cmd_force@std_msgs/msg/Float64[gz.msgs.Double]',
                        '/model/vibration_platform/joint/spring_rr/cmd_force@std_msgs/msg/Float64[gz.msgs.Double]',
                    ],
                    output='screen',
                    name='ros_gz_bridge'
                ),
            ]
        ),
    ])
