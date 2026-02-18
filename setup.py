from setuptools import find_packages, setup
import os
from glob import glob

package_name = "robot_sugar_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rebellion",
    maintainer_email="skarthikbe96@gmail.com",
    description="SUGAR ROS2 integration",
    license="TODO",
    entry_points={
        "console_scripts": [
            "sugar_ros_bridge = robot_sugar.sugar_ros_bridge:main",
            "sugar_perception_node = robot_sugar.perception.sugar_perception_node:main",
            "sugar_dataset_recorder = robot_sugar.dataset.sugar_dataset_recorder:main",
        ],
    },
)
