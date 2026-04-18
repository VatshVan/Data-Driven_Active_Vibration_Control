import os
from glob import glob
from setuptools import find_packages, setup

package_name = "vib_sim"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.py")),
        (os.path.join("share", package_name, "worlds"),
            glob("worlds/*.sdf")),
        (os.path.join("share", package_name, "config"),
            glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sahil",
    maintainer_email="sahil@todo.com",
    description="Spring Mass Damper Vibration Control with Neural MPC",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sensor_node = vib_sim.sensor_node:main",
            "disturbance_node = vib_sim.disturbance_node:main",
            "kalman_filter_node = vib_sim.kalman_filter_node:main",
            "data_collector_node = vib_sim.data_collector_node:main",
            "neural_controller_node = vib_sim.neural_controller_node:main",
            "platform_data_collector = vib_sim.platform_data_collector:main",
            "platform_disturbance = vib_sim.platform_disturbance:main",
            "platform_sensor = vib_sim.platform_sensor:main",
            "platform_neural_controller = vib_sim.platform_neural_controller:main",
            "platform_pid_controller = vib_sim.platform_pid_controller:main",
            "platform_kalman_filter = vib_sim.platform_kalman_filter:main",
            "evaluation_node = vib_sim.evaluation_node:main",
        ],
    },
)
