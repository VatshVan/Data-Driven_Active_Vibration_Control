import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('vib_sim')
    sdf_file = os.path.join(pkg_dir, 'worlds', 'spring_mass_damper.sdf')

    return LaunchDescription([

        # 1. Launch Gazebo with the SDF world
        ExecuteProcess(
            cmd=['gz', 'sim', sdf_file, '-r'],
            output='screen',
            name='gazebo'
        ),

        # 2. Bridge: Gazebo topics <-> ROS 2 topics
        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                        '/world/spring_mass_world/model/spring_mass_damper/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
                        '/model/spring_mass_damper/joint/spring_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
                    ],
                    output='screen',
                    name='ros_gz_bridge'
                ),
            ]
        ),
    ])
