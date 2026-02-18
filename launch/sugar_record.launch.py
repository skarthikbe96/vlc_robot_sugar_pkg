# launch/sugar_record.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    cfg = PathJoinSubstitution([FindPackageShare("robot_sugar"), "config", "sugar_record.yaml"])
    return LaunchDescription([
        Node(
            package="robot_sugar",
            executable="sugar_dataset_recorder",
            name="sugar_dataset_recorder",
            output="screen",
            parameters=[cfg],
        )
    ])
