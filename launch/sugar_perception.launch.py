# launch/sugar_perception.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    cfg = PathJoinSubstitution([FindPackageShare("robot_sugar"), "config", "sugar_perception.yaml"])
    return LaunchDescription([
        Node(
            package="robot_sugar",
            executable="sugar_perception_node",
            name="sugar_perception_node",
            output="screen",
            parameters=[cfg],
        )
    ])
