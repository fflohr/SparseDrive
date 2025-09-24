#ifndef SPARSEDrive_DISPLAY_HPP
#define SPARSEDrive_DISPLAY_HPP

#include <mutex>
#include <rviz_common/display.hpp>
#include <rviz_common/properties/ros_topic_property.hpp>
#include <rviz_common/properties/color_property.hpp>
#include <rviz_common/properties/float_property.hpp>

#include <vision_msgs/msg/detection3_d_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <multipath_msgs/msg/multi_path.hpp>

#include <rviz_rendering/objects/billboard_line.hpp>
#include <rviz_rendering/objects/shape.hpp>

namespace sparsedrive_rviz_plugin
{

class SparseDriveDisplay : public rviz_common::Display
{
  Q_OBJECT

public:
  SparseDriveDisplay();
  ~SparseDriveDisplay() override;

  void onInitialize() override;
  void onEnable() override;
  void onDisable() override;
  void update(float wall_dt, float ros_dt) override;
  void reset() override;

private Q_SLOTS:
  void updateTopics();

private:
  void detectionCallback(const vision_msgs::msg::Detection3DArray::ConstSharedPtr msg);
  void trajCallback(const multipath_msgs::msg::MultiPath::ConstSharedPtr msg);
  void planCallback(const nav_msgs::msg::Path::ConstSharedPtr msg);

  void processDetections();
  void processTrajectories();
  void processPlan();

  // RViz properties
  rviz_common::properties::RosTopicProperty<vision_msgs::msg::Detection3DArray> detection_topic_property_;
  rviz_common::properties::RosTopicProperty<multipath_msgs::msg::MultiPath> traj_topic_property_;
  rviz_common::properties::RosTopicProperty<nav_msgs::msg::Path> plan_topic_property_;
  rviz_common::properties::ColorProperty detection_color_property_;
  rviz_common::properties::ColorProperty traj_color_property_;
  rviz_common::properties::ColorProperty plan_color_property_;
  rviz_common::properties::FloatProperty line_width_property_;

  // ROS subscribers
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr detection_sub_;
  rclcpp::Subscription<multipath_msgs::msg::MultiPath>::SharedPtr traj_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr plan_sub_;

  // Message holders
  vision_msgs::msg::Detection3DArray::ConstSharedPtr detection_msg_;
  multipath_msgs::msg::MultiPath::ConstSharedPtr traj_msg_;
  nav_msgs::msg::Path::ConstSharedPtr plan_msg_;

  // Scene objects
  std::vector<std::unique_ptr<rviz_rendering::Shape>> detection_visuals_;
  std::vector<std::unique_ptr<rviz_rendering::BillboardLine>> traj_visuals_;
  std::unique_ptr<rviz_rendering::BillboardLine> plan_visual_;

  std::mutex mutex_;
};

}  // namespace sparsedrive_rviz_plugin

#endif  // SPARSEDrive_DISPLAY_HPP