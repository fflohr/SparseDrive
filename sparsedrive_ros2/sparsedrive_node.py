import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import ReliabilityPolicy, QoSProfile, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from multipath_msgs.msg import MultiPath
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
import cv2
from mmdet.datasets.pipelines import to_tensor, Compose
from mmcv.parallel import DataContainer as DC
import sys
sys.path.insert(0, '/home/kanak/SparseDrive')
from projects.mmdet3d_plugin.datasets.pipelines import *
import torch
import numpy as np
import time, json, torch
from pyquaternion import Quaternion
import copy
import PIL
from pytictoc import TicToc
from .utils import CLASSES, MAP_CLASSES, ResizeCropFlipImage, format_transforms, init_data, test_pipeline, get_model, get_data, get_augmentation, format_data

COLOR_VECTORS = [(100, 149, 237), (70, 130, 180), (119, 136, 153)] # RGB

class SparseDriveNode(Node):
    def __init__(self):
        super().__init__("SparseDriveNode")
        self.declare_parameter("config", "/home/kanak/SparseDrive/projects/configs/sparsedrive_small_stage2.py")
        self.config = self.get_parameter("config").get_parameter_value().string_value
        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.declare_parameter("score_threshold", 0.3)
        self.score_threshold = self.get_parameter("score_threshold").get_parameter_value().double_value
        self.group = ReentrantCallbackGroup()
        self.qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.cv_bridge = CvBridge()
        self.model = get_model(self.config, self.device)
        self.pipeline = Compose(test_pipeline)
        self.results = init_data(1200, 2200)
        self.color_image_msg = None
        self.odom_msg = None
        self.camera_intrinsics = None
        self.ego2global = None
        self.timestamp = None
        self.output = None
        self.LIDAR_TOP_FRAME_ID = 'os_lidar_roof_top'
        self.lidar2ego, self.lidar2cam, self.lidar2img = format_transforms()
        self._det_pub = self.create_publisher(Detection3DArray, "detection", 10)
        self._traj_pub = self.create_publisher(MultiPath, "traj", 10)
        self._plan_pub = self.create_publisher(Path, "plan", 10)
        self._plan_gt_pub = self.create_publisher(Path, "plan_gt", 10)
        self._map_pred_pub = self.create_publisher(MarkerArray, "map_pred", 10)
        self._map_gt_pub = self.create_publisher(MarkerArray, "map_gt", 10)
        self.create_subscription(Image, "image_raw", self.color_image_callback, self.qos_profile, callback_group=self.group)
        self.create_subscription(CameraInfo, 'camera_info', self.camera_info_callback, QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE), callback_group=self.group)
        self.create_subscription(Odometry, "odom", self.odom_callback, self.qos_profile, callback_group=self.group)
        self.create_timer(0.01, self.predict, callback_group=self.group)

    def color_image_callback(self, msg): self.color_image_msg = msg
    def camera_info_callback(self, msg):
        if self.camera_intrinsics is None: self.camera_intrinsics = np.array(msg.k).reshape(3, 3)
    def odom_callback(self, msg):
        self.odom_msg = msg
        pos = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        q = Quaternion(w=orientation.w, x=orientation.x, y=orientation.y, z=orientation.z)
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = q.rotation_matrix
        pose_matrix[:3, 3] = [pos.x, pos.y, pos.z]
        self.ego2global = pose_matrix
        t = msg.header.stamp
        self.timestamp = t.sec + t.nanosec * 1e-9

    def predict(self):
        if self.color_image_msg is not None:
            cv_color_image = self.cv_bridge.imgmsg_to_cv2(self.color_image_msg, desired_encoding='bayer_rggb8')
            cv_color_image = cv2.cvtColor(cv_color_image, cv2.COLOR_BayerRG2RGB)
            data = get_data(self.results.copy(), cv_color_image, self.camera_intrinsics, self.ego2global, self.lidar2ego, self.lidar2cam, self.lidar2img, self.timestamp)
            info = self.pipeline(data)
            imgs, info = format_data(info)
            with torch.no_grad(): self.output = self.model(imgs, **info)
            self.visualize()

    def visualize(self):
        if self.output is not None:
            output_data = self.output[0]['img_bbox']
            self.publish_detections(output_data)
            self.publish_trajectories(output_data)
            self.publish_plan(output_data, 'final_planning', self._plan_pub)
            if 'planning_gt' in output_data:
                self.publish_plan(output_data, 'planning_gt', self._plan_gt_pub)
            self.publish_map(output_data)

    def publish_detections(self, data):
        det_msg = Detection3DArray()
        det_msg.header.frame_id = self.LIDAR_TOP_FRAME_ID
        det_msg.header.stamp = self.odom_msg.header.stamp
        for i, score in enumerate(data['scores_3d'].cpu().numpy()):
            if score > self.score_threshold:
                det = Detection3D()
                det.header = det_msg.header
                bbox = data['boxes_3d'][i]
                det.bbox.center.position.x = float(bbox[0])
                det.bbox.center.position.y = float(bbox[1])
                det.bbox.center.position.z = float(bbox[2])
                det.bbox.size.x = float(bbox[3])
                det.bbox.size.y = float(bbox[4])
                det.bbox.size.z = float(bbox[5])
                q = Quaternion(axis=[0, 0, 1], radians=bbox[6])
                det.bbox.center.orientation.x, det.bbox.center.orientation.y, det.bbox.center.orientation.z, det.bbox.center.orientation.w = q.x, q.y, q.z, q.w
                det_msg.detections.append(det)
        self._det_pub.publish(det_msg)

    def publish_trajectories(self, data):
        traj_msg = MultiPath()
        for i, score in enumerate(data['scores_3d'].cpu().numpy()):
            if score > self.score_threshold:
                traj_path = Path()
                traj_path.header.stamp = self.get_clock().now().to_msg()
                traj_path.header.frame_id = 'base_link'
                for waypoint in data['trajs_3d'][i][0]:
                    pose = PoseStamped()
                    pose.pose.position.x, pose.pose.position.y = float(waypoint[1]), float(waypoint[0])
                    traj_path.poses.append(pose)
                traj_msg.paths.append(traj_path)
        self._traj_pub.publish(traj_msg)

    def publish_plan(self, data, key, pub):
        plan_msg = Path()
        plan_msg.header.stamp = self.get_clock().now().to_msg()
        plan_msg.header.frame_id = 'os_sensor_roof_top'
        for waypoint in data[key].cpu().numpy():
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y = float(waypoint[1]), -float(waypoint[0])
            plan_msg.poses.append(pose)
        pub.publish(plan_msg)

    def publish_map(self, data):
        if 'vectors' in data:
            self._map_pred_pub.publish(self.create_marker_array(data, 'map_pred'))
        if 'map_gt' in data:
            self._map_gt_pub.publish(self.create_marker_array(data['map_gt'], 'map_gt'))

    def create_marker_array(self, map_data, ns):
        marker_array = MarkerArray()
        vectors = map_data['vectors']
        labels = map_data.get('labels', [0] * len(vectors))
        scores = map_data.get('scores', [1.0] * len(vectors))
        for i, (vec, label, score) in enumerate(zip(vectors, labels, scores)):
            marker = Marker()
            marker.header.frame_id = "os_sensor_roof_top"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.2
            color = COLOR_VECTORS[label % len(COLOR_VECTORS)]
            marker.color = ColorRGBA(r=color[0]/255.0, g=color[1]/255.0, b=color[2]/255.0, a=1.0)
            marker.text = f"score: {score:.2f}"
            for point in vec:
                p = Point()
                p.x, p.y = float(point[0]), float(point[1])
                marker.points.append(p)
            marker_array.markers.append(marker)
        return marker_array

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(SparseDriveNode())
    rclpy.shutdown()

if __name__ == "__main__":
    main()