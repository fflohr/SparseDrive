import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray
from nav_msgs.msg import Path
from multipath_msgs.msg import MultiPath
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
import numpy as np
import cv2
from pyquaternion import Quaternion
from ament_index_python.packages import get_package_share_directory
import os

# Constants from original bev_render.py, converted to BGR for OpenCV
COLOR_MAPPING = np.array([
    [0, 0, 0], [0, 179, 255], [117, 62, 128], [0, 104, 255], [215, 189, 166],
    [32, 0, 193], [98, 162, 206], [102, 112, 129], [52, 125, 0], [142, 118, 246],
    [138, 83, 0], [92, 122, 255], [122, 55, 83], [0, 142, 255], [81, 40, 179],
    [0, 200, 244], [13, 24, 127], [0, 170, 147], [21, 51, 89], [19, 58, 241],
    [22, 44, 35]
], dtype=np.uint8)

class BEVRendererNode(Node):
    def __init__(self):
        super().__init__('bev_renderer_node')
        self.xlim = 40
        self.ylim = 40
        self.image_width = 800
        self.image_height = 800
        self.scale = self.image_width / (2 * self.xlim)

        self.cv_bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, '/bev_image', 10)

        # Data storage
        self.latest_det = None
        self.latest_traj = None
        self.latest_plan = None
        self.latest_plan_gt = None
        self.latest_map_pred = None
        self.latest_map_gt = None

        # Subscribers
        self.create_subscription(Detection3DArray, '/detection', self.det_callback, 10)
        self.create_subscription(MultiPath, '/traj', self.traj_callback, 10)
        self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.create_subscription(Path, '/plan_gt', self.plan_gt_callback, 10)
        self.create_subscription(MarkerArray, '/map_pred', self.map_pred_callback, 10)
        self.create_subscription(MarkerArray, '/map_gt', self.map_gt_callback, 10)

        # Trigger
        self.create_subscription(Image, '/image_raw', self.render_trigger, 10)

    def det_callback(self, msg): self.latest_det = msg
    def traj_callback(self, msg): self.latest_traj = msg
    def plan_callback(self, msg): self.latest_plan = msg
    def plan_gt_callback(self, msg): self.latest_plan_gt = msg
    def map_pred_callback(self, msg): self.latest_map_pred = msg
    def map_gt_callback(self, msg): self.latest_map_gt = msg

    def render_trigger(self, msg):
        bev_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        if self.latest_map_gt: self.draw_map(bev_image, self.latest_map_gt)
        if self.latest_map_pred: self.draw_map(bev_image, self.latest_map_pred)
        if self.latest_det: self.draw_detection_pred(bev_image, self.latest_det)
        if self.latest_traj: self.draw_motion_pred(bev_image, self.latest_traj)
        if self.latest_plan: self.draw_planning(bev_image, self.latest_plan, 'autumn')
        if self.latest_plan_gt: self.draw_planning(bev_image, self.latest_plan_gt, 'winter', linestyle='--')

        self.render_sdc_car(bev_image)
        self.render_legend(bev_image)

        img_msg = self.cv_bridge.cv2_to_imgmsg(bev_image, "bgr8")
        img_msg.header = msg.header
        self.image_publisher.publish(img_msg)

    def world_to_pixel(self, x, y):
        return int(-y * self.scale + self.image_width / 2), int(-x * self.scale + self.image_height / 2)

    def draw_detection_pred(self, image, msg):
        for i, det in enumerate(msg.detections):
            color = COLOR_MAPPING[(i + 1) % len(COLOR_MAPPING)].tolist()
            center_x, center_y = det.bbox.center.position.x, det.bbox.center.position.y
            q = Quaternion(w=det.bbox.center.orientation.w, x=det.bbox.center.orientation.x, y=det.bbox.center.orientation.y, z=det.bbox.center.orientation.z)
            yaw = q.yaw_pitch_roll[0]
            width, length = det.bbox.size.x * self.scale, det.bbox.size.y * self.scale

            box_center_px = self.world_to_pixel(center_x, center_y)
            box = cv2.boxPoints(((box_center_px[0], box_center_px[1]), (width, length), -np.degrees(yaw)))
            box = np.int0(box)
            cv2.drawContours(image, [box], 0, color, 2)
            forward_center = (np.mean(box[:2, 0]), np.mean(box[:2, 1]))
            cv2.line(image, (box_center_px[0], box_center_px[1]), (int(forward_center[0]), int(forward_center[1])), color, 2)

    def draw_motion_pred(self, image, msg):
        for i, path in enumerate(msg.paths):
            color = COLOR_MAPPING[(i + 1) % len(COLOR_MAPPING)].tolist()
            points = np.array([self.world_to_pixel(p.pose.position.x, p.pose.position.y) for p in path.poses], dtype=np.int32)
            self.render_traj(image, points, 'winter')

    def draw_planning(self, image, msg, colormap, linestyle='-'):
        points = np.array([self.world_to_pixel(p.pose.position.x, p.pose.position.y) for p in msg.poses], dtype=np.int32)
        self.render_traj(image, points, colormap, linestyle)

    def draw_map(self, image, msg):
        for marker in msg.markers:
            points = []
            for p in marker.points:
                points.append(self.world_to_pixel(p.x, p.y))

            if len(points) > 1:
                color = (marker.color.b * 255, marker.color.g * 255, marker.color.r * 255) # BGR
                cv2.polylines(image, [np.array(points, dtype=np.int32)], isClosed=False, color=color, thickness=2)

    def render_traj(self, image, traj, colormap, linestyle='-'):
        cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1), getattr(cv2, f'COLORMAP_{colormap.upper()}'))
        for i in range(len(traj) - 1):
            color_index = int((i / (len(traj) - 1)) * 255) if len(traj) > 1 else 0
            color = cmap[0, color_index].tolist()

            if linestyle == '-':
                cv2.line(image, tuple(traj[i]), tuple(traj[i+1]), color, 2)
            elif linestyle == '--':
                cv2.line(image, tuple(traj[i]), tuple(traj[i+1]), color, 1, cv2.LINE_AA)

    def render_sdc_car(self, image):
        try:
            share_dir = get_package_share_directory('sparsedrive_ros2')
            car_path = os.path.join(share_dir, 'resource', 'sdc_car.png')
            sdc_car_png = cv2.imread(car_path, cv2.IMREAD_UNCHANGED)
            sdc_car_png = cv2.resize(sdc_car_png, (int(2 * self.scale), int(4 * self.scale)))
            h, w, _ = sdc_car_png.shape
            x_offset = int(self.image_width / 2 - w / 2)
            y_offset = int(self.image_height / 2 - h / 2)

            alpha_s = sdc_car_png[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                image[y_offset:y_offset+h, x_offset:x_offset+w, c] = (alpha_s * sdc_car_png[:, :, c] +
                                                                      alpha_l * image[y_offset:y_offset+h, x_offset:x_offset+w, c])
        except Exception as e:
            self.get_logger().warn(f"Error rendering SDC car: {e}")

    def render_legend(self, image):
        try:
            share_dir = get_package_share_directory('sparsedrive_ros2')
            legend_path = os.path.join(share_dir, 'resource', 'legend.png')
            legend_png = cv2.imread(legend_path)
            h, w, _ = legend_png.shape
            x_offset = self.image_width - w - 10
            y_offset = self.image_height - h - 10
            image[y_offset:y_offset+h, x_offset:x_offset+w] = legend_png
        except Exception as e:
            self.get_logger().warn(f"Error rendering legend: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BEVRendererNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()