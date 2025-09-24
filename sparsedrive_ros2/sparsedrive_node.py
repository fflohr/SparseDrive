# Basic ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import ReliabilityPolicy, QoSProfile, HistoryPolicy

# Executor and callback imports
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor

# ROS2 interfaces
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import Detection3D
from vision_msgs.msg import ObjectHypothesisWithPose
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from multipath_msgs.msg import MultiPath

from message_filters import Subscriber, ApproximateTimeSynchronizer

# Image msg parser
from cv_bridge import CvBridge
import cv2

# MMCV
from mmdet.datasets.pipelines import to_tensor, Compose
from mmcv.parallel import DataContainer as DC

import sys
sys.path.insert(0, '/home/kanak/SparseDrive')
from projects.mmdet3d_plugin.datasets.pipelines import *

import torch

# Others
import numpy as np
import time, json, torch
from pyquaternion import Quaternion
import copy
import PIL

from pytictoc import TicToc

from .utils import CLASSES, MAP_CLASSES, ResizeCropFlipImage, format_transforms, init_data, test_pipeline, get_model, init_data, get_data, get_augmentation, format_data, format_transforms

class SparseDriveNode(Node):
    
    def __init__(self):
        super().__init__("SparseDriveNode")
        
        ## Declare parameters for node
        self.declare_parameter("config", "/home/kanak/SparseDrive/projects/configs/sparsedrive_small_stage2.py")
        self.config = self.get_parameter("config").get_parameter_value().string_value
        
        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter("device").get_parameter_value().string_value
        
        self.declare_parameter("score_threshold", 0.3)
        self.score_threshold = self.get_parameter("score_threshold").get_parameter_value().double_value

        ## other inits
        self.group_1 = MutuallyExclusiveCallbackGroup() # camera subscribers
        self.group_2 = MutuallyExclusiveCallbackGroup() # predict timer
        self.group_3 = MutuallyExclusiveCallbackGroup() # predict timer
        self.group = ReentrantCallbackGroup()
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.cv_bridge = CvBridge()

        self.original_img_shape = (1200, 2200)
        self.img_shape = (384, 704)
        self.scale_factor = min(self.img_shape[0] / self.original_img_shape[0],
                                        self.img_shape[1] / self.original_img_shape[1])
        self.resize_crop_filter = ResizeCropFlipImage()

        # SparseDrive
        self.model = get_model(self.config, self.device)
        self.pipeline = Compose(test_pipeline)
        self.results = init_data(self.original_img_shape[0], self.original_img_shape[1])

        self.color_image_msg = None
        self.odom_msg = None
        self.camera_intrinsics = None
        self.ego2global = None
        self.timestamp = None
        # self.pred_image_msg = Image()
        self.output = None
        self.LIDAR_TOP_FRAME_ID = 'os_lidar_roof_top'


        self.lidar2ego, self.lidar2cam, self.lidar2img = format_transforms()
        
        # Publishers
        # self._pred_pub = self.create_publisher(Image, "/prediction/image", 10)
        self._det_pub = self.create_publisher(Detection3DArray,
                                                 "detection",
                                                 10)
        self._traj_pub = self.create_publisher(MultiPath,
                                                 "traj",
                                                 10)
        self._plan_pub = self.create_publisher(Path,
                                                 "plan",
                                                 10)
        self._plan_gt_pub = self.create_publisher(Path, "plan_gt", 10)
        self._map_pred_pub = self.create_publisher(Image, "map_pred", 10)
        self._map_gt_pub = self.create_publisher(Image, "map_gt", 10)
        
        # Subscribers
        self._color_image_sub = self.create_subscription(Image, "image_raw", self.color_image_callback, self.qos_profile, callback_group=self.group_1)
        self._camera_info_subscriber = self.create_subscription(CameraInfo, 'camera_info', self.camera_info_callback, QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE), callback_group=self.group_1)
        self._odom_sub = self.create_subscription(Odometry, "odom", self.odom_callback, self.qos_profile, callback_group=self.group_1)
        
        # Timers
        self._inference_timer = self.create_timer(0.01, self.predict, callback_group=self.group)
        # self.visualization_timer = self.create_timer(0.1, self.visualize, callback_group=self.group_2) # 10 hz
        self.t = TicToc()
    
    def color_image_callback(self, msg):
        self.color_image_msg = msg
    
    def camera_info_callback(self, msg):
        try:
            if self.camera_intrinsics is None:
                self.camera_intrinsics = np.array(msg.k).reshape(3, 3)
                self.get_logger().info('Camera intrinsics have been set!')
            
        except Exception as e:
            self.get_logger().error(f'camera_info_callback Error: {e}')
    
    def odom_callback(self, msg):
        self.odom_msg = msg
        pos = msg.pose.pose.position
        x, y, z = pos.x, pos.y, pos.z

        orientation = msg.pose.pose.orientation
        q = Quaternion(w=orientation.w, x=orientation.x, y=orientation.y, z=orientation.z)

        rot_matrix = q.rotation_matrix  # shape (3, 3)

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rot_matrix
        pose_matrix[:3, 3] = [x, y, z]

        self.ego2global = pose_matrix
        t = msg.header.stamp
        self.timestamp = t.sec + t.nanosec * 1e-9
    
    def predict(self):
        if self.color_image_msg is not None:      
            # Convert color image msg
            self.t.tic()
            cv_color_image = self.cv_bridge.imgmsg_to_cv2(self.color_image_msg, desired_encoding='bayer_rggb8')
            cv_color_image = cv2.cvtColor(cv_color_image, cv2.COLOR_BayerRG2RGB) 
            np_color_image = np.array(cv_color_image, dtype=np.uint8)

            # Format
            # results = copy.deepcopy(self.results)
            results = self.results.copy()
            # self.t.toc('copy')

            results['already_resized'] = False

            # Resizing
            resize = results['aug_config'].get("resize", 1)
            resize_dims = results['aug_config'].get("resize_dims")
            crop = results['aug_config'].get("crop", [0, 0, *resize_dims])
            img = PIL.Image.fromarray(np_color_image)
            np_color_image = np.array(img.resize(resize_dims).crop(crop))
            results['aug_config']['already_resized'] = True
            
            # # Raw Bayer image data (8-bit)
            # raw_data = np.frombuffer(self.color_image_msg.data, dtype=np.uint8)
            # # Reshape to 2D image
            # bayer_image = raw_data.reshape((height, width))
            # # Debayer (e.g., RGGB)
            # bgr_image = cv2.cvtColor(bayer_image, cv2.COLOR_BayerRG2BGR)

            data = get_data(results, np_color_image, self.camera_intrinsics, self.ego2global, self.lidar2ego, self.lidar2cam, self.lidar2img, self.timestamp)
            # self.t.toc('data pre-processing ')

            data = self.resize_crop_filter(data)

            info = self.pipeline(data)
            # self.t.toc('data processing ')

            imgs, info = format_data(info)
            # self.t.toc('data formatting ')

            # Inference
            with torch.no_grad():
                self.output = self.model(imgs, **info)
            # self.t.toc('inference')
            self.visualize()
            self.t.toc(restart=True)
    
    def visualize(self):
        det_msg = Detection3DArray()
        plan_msg = Path()
        traj_msg = MultiPath()

        if self.output is not None:
            scores = self.output[0]['img_bbox']['scores_3d'].cpu().numpy()
            bboxes = self.output[0]['img_bbox']['boxes_3d'].cpu().numpy()
            labels = self.output[0]['img_bbox']['labels_3d'].cpu().numpy()
            trajs_scores = self.output[0]['img_bbox']['trajs_score'].cpu().numpy()
            trajs = self.output[0]['img_bbox']['trajs_3d'].cpu().numpy()
            plan = self.output[0]['img_bbox']['final_planning'].cpu().numpy()
            
            if scores.size != 0:
                for i, score in enumerate(scores[:10]):
                    if(score > self.score_threshold):
                        det = Detection3D()
                        traj_path = Path()
                        det.header.frame_id = self.LIDAR_TOP_FRAME_ID
                        det.header.stamp = self.odom_msg.header.stamp
                        traj_path.header.stamp = self.get_clock().now().to_msg()
                        traj_path.header.frame_id = 'base_link'
                        center = bboxes[i, 0 : 3]
                        box_dims = bboxes[i, 3 : 6]
                        quat = Quaternion(axis=[0, 0, 1], radians=bboxes[i][6])
                        det.bbox.center.orientation.x = quat[1]
                        det.bbox.center.orientation.y = quat[2]
                        det.bbox.center.orientation.z = quat[3]
                        det.bbox.center.orientation.w = quat[0]
                        det.bbox.center.position.x = float(bboxes[i][0])
                        det.bbox.center.position.y = float(bboxes[i][1])
                        det.bbox.center.position.z = float(bboxes[i][2]) 
                        det.bbox.size.x = float(bboxes[i][3])
                        det.bbox.size.y = float(bboxes[i][4])
                        det.bbox.size.z = float(bboxes[i][5])
                        hypothesis = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.class_id = str(labels[i])
                        hypothesis.hypothesis.score = float(score)
                        hypothesis.pose.pose = det.bbox.center
                        det.id = str(labels[i])
                        det.results.append(hypothesis)
                        det_msg.detections.append(det)

                        for i, waypoint in enumerate(trajs[i][0]):
                            pose = PoseStamped()
                            pose.header.stamp = self.get_clock().now().to_msg()
                            pose.header.frame_id = 'base_link'
                            pose.pose.position.x = float(waypoint[1])
                            pose.pose.position.y = float(waypoint[0])
                            pose.pose.position.z = 0.0
                            pose.pose.orientation.w = 1.0  # Facing forward (identity quaternion)

                            traj_path.poses.append(pose)
                    
                        traj_msg.paths.append(traj_path)
        
            det_msg.header.frame_id = self.LIDAR_TOP_FRAME_ID
            det_msg.header.stamp = self.odom_msg.header.stamp 

            if len(det_msg.detections) != 0:
                self._det_pub.publish(det_msg)
                det_msg.detections = []
            else:
                det_msg.detections = []
                self._det_pub.publish(det_msg)
            
            if len(traj_msg.paths) != 0:
                self._traj_pub.publish(traj_msg)
                traj_msg.paths = []
            else:
                traj_msg.paths = []
                self._traj_pub.publish(traj_msg)

            for i, waypoint in enumerate(plan):
                pose = PoseStamped()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = 'os_sensor_roof_top'
                pose.pose.position.x = float(waypoint[1])
                pose.pose.position.y = -float(waypoint[0])
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0  # Facing forward (identity quaternion)

                plan_msg.poses.append(pose)
            
            plan_msg.header.stamp = self.get_clock().now().to_msg()
            plan_msg.header.frame_id = 'os_sensor_roof_top'

            if len(plan_msg.poses) != 0:
                self._plan_pub.publish(plan_msg)
                plan_msg.poses = []
            else:
                plan_msg.poses = []
                self._plan_pub.publish(plan_msg)

            # Publish map and ground truth data if available
            if 'map_pred' in self.output[0]['img_bbox']:
                map_pred = self.output[0]['img_bbox']['map_pred'].cpu().numpy()
                map_pred_msg = self.cv_bridge.cv2_to_imgmsg(map_pred, "mono8")
                map_pred_msg.header.stamp = self.get_clock().now().to_msg()
                self._map_pred_pub.publish(map_pred_msg)

            if 'map_gt' in self.output[0]['img_bbox']:
                map_gt = self.output[0]['img_bbox']['map_gt'].cpu().numpy()
                map_gt_msg = self.cv_bridge.cv2_to_imgmsg(map_gt, "mono8")
                map_gt_msg.header.stamp = self.get_clock().now().to_msg()
                self._map_gt_pub.publish(map_gt_msg)

            if 'planning_gt' in self.output[0]['img_bbox']:
                plan_gt = self.output[0]['img_bbox']['planning_gt'].cpu().numpy()
                plan_gt_msg = Path()
                for i, waypoint in enumerate(plan_gt):
                    pose = PoseStamped()
                    pose.header.stamp = self.get_clock().now().to_msg()
                    pose.header.frame_id = 'os_sensor_roof_top'
                    pose.pose.position.x = float(waypoint[1])
                    pose.pose.position.y = -float(waypoint[0])
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.w = 1.0
                    plan_gt_msg.poses.append(pose)
                plan_gt_msg.header.stamp = self.get_clock().now().to_msg()
                plan_gt_msg.header.frame_id = 'os_sensor_roof_top'
                self._plan_gt_pub.publish(plan_gt_msg)
                
    def shutdown_callback(self):
        self.get_logger().warn("Shutting down...")
        
        

def main(args=None):
    rclpy.init(args=args)

    # Instansiate node class
    vision_node = SparseDriveNode()

    # Create executor
    # executor = MultiThreadedExecutor()
    executor = SingleThreadedExecutor()
    executor.add_node(vision_node)
    
    try:
        # Run executor
        executor.spin()
        
    except KeyboardInterrupt:
        pass
    
    finally:
        # Shutdown executor
        vision_node.shutdown_callback()
        executor.shutdown()

if __name__ == "__main__":
    main()