import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray
from nav_msgs.msg import Path
from multipath_msgs.msg import MultiPath
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer, Cache

from .drawing_utils import draw_bev

class BEVVisualizer(Node):
    def __init__(self):
        super().__init__('bev_visualizer')

        self.declare_parameter('image_width', 800)
        self.declare_parameter('image_height', 800)
        self.declare_parameter('resolution', 0.1)

        self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value
        self.resolution = self.get_parameter('resolution').get_parameter_value().double_value

        self.cv_bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, '/bev_image', 10)

        # Core subscribers
        self.detection_sub = Subscriber(self, Detection3DArray, '/detection')
        self.traj_sub = Subscriber(self, MultiPath, '/traj')
        self.plan_sub = Subscriber(self, Path, '/plan')

        # Optional subscribers with caches
        self.plan_gt_sub = Subscriber(self, Path, '/plan_gt')
        self.plan_gt_cache = Cache(self.plan_gt_sub, 1)
        self.map_pred_sub = Subscriber(self, Image, '/map_pred')
        self.map_pred_cache = Cache(self.map_pred_sub, 1)
        self.map_gt_sub = Subscriber(self, Image, '/map_gt')
        self.map_gt_cache = Cache(self.map_gt_sub, 1)

        self.ts = ApproximateTimeSynchronizer(
            [self.detection_sub, self.traj_sub, self.plan_sub],
            queue_size=10,
            slop=0.1)
        self.ts.registerCallback(self.visualize_callback)

    def visualize_callback(self, detection_msg, traj_msg, plan_msg):
        timestamp = detection_msg.header.stamp

        # Get optional messages from cache
        plan_gt_msg = self.plan_gt_cache.getElemBeforeOrAt(timestamp)
        map_pred_msg = self.map_pred_cache.getElemBeforeOrAt(timestamp)
        map_gt_msg = self.map_gt_cache.getElemBeforeOrAt(timestamp)

        bev_image = draw_bev(
            detection_msg,
            traj_msg,
            plan_msg,
            self.image_width,
            self.image_height,
            self.resolution,
            map_pred_msg,
            map_gt_msg,
            plan_gt_msg
        )

        image_msg = self.cv_bridge.cv2_to_imgmsg(bev_image, "bgr8")
        image_msg.header = detection_msg.header
        self.image_publisher.publish(image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BEVVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()