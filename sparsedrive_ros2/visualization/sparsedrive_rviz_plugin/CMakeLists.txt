cmake_minimum_required(VERSION 3.5)
project(sparsedrive_rviz_plugin)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(multipath_msgs REQUIRED)

# Add source files for the plugin
add_library(sparsedrive_rviz_plugin src/sparsedrive_display.cpp)

# Link against required libraries
ament_target_dependencies(sparsedrive_rviz_plugin
  "rclcpp"
  "rviz_common"
  "vision_msgs"
  "nav_msgs"
  "multipath_msgs")

# Install the plugin library
install(TARGETS sparsedrive_rviz_plugin
  DESTINATION lib/${PROJECT_NAME})

# Install the plugin description file
install(FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME})

# Export the plugin description file
pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

ament_package()