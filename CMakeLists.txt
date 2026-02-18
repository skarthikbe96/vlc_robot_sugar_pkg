cmake_minimum_required(VERSION 3.16)
project(robot_sugar_pkg)

find_package(ament_cmake REQUIRED)

# Install your Python nodes as executables.
# Make sure these files have a shebang (#!/usr/bin/env python3) and +x perms.
install(PROGRAMS
  robot_sugar/sugar_ros_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME sugar_ros_bridge
)

# (Optional) also install your helper scripts so you can ros2 run them directly
install(PROGRAMS
  robot_sugar/sugar_infer_standalone.py
  robot_sugar/dataset/sugar_dataset_recorder.py
  robot_sugar/perception/sugar_perception_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
