#include "sparsedrive_display.hpp"

#include <rviz_common/properties/property.hpp>
#include <rviz_rendering/objects/shape.hpp>
#include <rviz_rendering/objects/billboard_line.hpp>

namespace sparsedrive_rviz_plugin
{

SparseDriveDisplay::SparseDriveDisplay()
: detection_topic_property_("Detection Topic", "/detection", "Topic for Detection3DArray messages", this, SLOT(updateTopics())),
  traj_topic_property_("Trajectory Topic", "/traj", "Topic for MultiPath messages", this, SLOT(updateTopics())),
  plan_topic_property_("Plan Topic", "/plan", "Topic for Path messages", this, SLOT(updateTopics())),
  detection_color_property_("Detection Color", QColor(0, 255, 0), "Color for detections", this),
  traj_color_property_("Trajectory Color", QColor(255, 0, 0), "Color for trajectories", this),
  plan_color_property_("Plan Color", QColor(0, 0, 255), "Color for the plan", this),
  line_width_property_("Line Width", 0.1, "Line width for trajectories and plan", this)
{
}

SparseDriveDisplay::~SparseDriveDisplay() = default;

void SparseDriveDisplay::onInitialize()
{
  Display::onInitialize();
  updateTopics();
}

void SparseDriveDisplay::onEnable()
{
  // Subscribe to topics
  updateTopics();
}

void SparseDriveDisplay::onDisable()
{
  // Unsubscribe from topics
  detection_sub_.reset();
  traj_sub_.reset();
  plan_sub_.reset();
}

void SparseDriveDisplay::updateTopics()
{
  // Unsubscribe from old topics
  detection_sub_.reset();
  traj_sub_.reset();
  plan_sub_.reset();

  // Subscribe to new topics
  detection_sub_ = getRosNode()->create_subscription<vision_msgs::msg::Detection3DArray>(
    detection_topic_property_.getTopicStd(), 10,
    std::bind(&SparseDriveDisplay::detectionCallback, this, std::placeholders::_1));
  traj_sub_ = getRosNode()->create_subscription<multipath_msgs::msg::MultiPath>(
    traj_topic_property_.getTopicStd(), 10,
    std::bind(&SparseDriveDisplay::trajCallback, this, std::placeholders::_1));
  plan_sub_ = getRosNode()->create_subscription<nav_msgs::msg::Path>(
    plan_topic_property_.getTopicStd(), 10,
    std::bind(&SparseDriveDisplay::planCallback, this, std::placeholders::_1));
}

void SparseDriveDisplay::update(float wall_dt, float ros_dt)
{
  std::lock_guard<std::mutex> lock(mutex_);
  processDetections();
  processTrajectories();
  processPlan();
}

void SparseDriveDisplay::reset()
{
  Display::reset();
  detection_visuals_.clear();
  traj_visuals_.clear();
  plan_visual_.reset();
}

void SparseDriveDisplay::detectionCallback(const vision_msgs::msg::Detection3DArray::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  detection_msg_ = msg;
}

void SparseDriveDisplay::trajCallback(const multipath_msgs::msg::MultiPath::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  traj_msg_ = msg;
}

void SparseDriveDisplay::planCallback(const nav_msgs::msg::Path::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  plan_msg_ = msg;
}

void SparseDriveDisplay::processDetections()
{
  if (!detection_msg_) return;

  detection_visuals_.clear();

  for (const auto & detection : detection_msg_->detections) {
    auto shape = std::make_unique<rviz_rendering::Shape>(
      rviz_rendering::Shape::Cube, context_->getSceneManager(), scene_node_);
    shape->setPosition(Ogre::Vector3(
      detection.bbox.center.position.x,
      detection.bbox.center.position.y,
      detection.bbox.center.position.z));
    shape->setOrientation(Ogre::Quaternion(
      detection.bbox.center.orientation.w,
      detection.bbox.center.orientation.x,
      detection.bbox.center.orientation.y,
      detection.bbox.center.orientation.z));
    shape->setScale(Ogre::Vector3(
      detection.bbox.size.x,
      detection.bbox.size.y,
      detection.bbox.size.z));

    Ogre::ColourValue color = detection_color_property_.getOgreColor();
    shape->setColor(color);
    detection_visuals_.push_back(std::move(shape));
  }
  detection_msg_.reset();
}

void SparseDriveDisplay::processTrajectories()
{
  if (!traj_msg_) return;

  traj_visuals_.clear();

  for (const auto & path : traj_msg_->paths) {
    auto line = std::make_unique<rviz_rendering::BillboardLine>(context_->getSceneManager(), scene_node_);
    line->setLineWidth(line_width_property_.getFloat());
    line->setColor(traj_color_property_.getOgreColor());

    for (const auto & pose : path.poses) {
      line->addPoint(Ogre::Vector3(
        pose.pose.position.x,
        pose.pose.position.y,
        pose.pose.position.z));
    }
    traj_visuals_.push_back(std::move(line));
  }
  traj_msg_.reset();
}

void SparseDriveDisplay::processPlan()
{
  if (!plan_msg_) return;

  plan_visual_.reset();
  auto line = std::make_unique<rviz_rendering::BillboardLine>(context_->getSceneManager(), scene_node_);
  line->setLineWidth(line_width_property_.getFloat());
  line->setColor(plan_color_property_.getOgreColor());

  for (const auto & pose : plan_msg_->poses) {
    line->addPoint(Ogre::Vector3(
      pose.pose.position.x,
      pose.pose.position.y,
      pose.pose.position.z));
  }
  plan_visual_ = std::move(line);
  plan_msg_.reset();
}

}  // namespace sparsedrive_rviz_plugin

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(sparsedrive_rviz_plugin::SparseDriveDisplay, rviz_common::Display)