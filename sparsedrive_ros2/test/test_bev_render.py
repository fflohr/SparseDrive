import unittest
import numpy as np
import cv2
from sparsedrive_ros2.visualization.bev_render import BEVRendererNode, COLOR_MAPPING
from vision_msgs.msg import Detection3DArray, Detection3D
from nav_msgs.msg import Path
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
import rclpy

class TestBEVRendererLogic(unittest.TestCase):

    def setUp(self):
        rclpy.init()
        self.node = BEVRendererNode()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_rendering_logic(self):
        bev_image = np.zeros((self.node.image_height, self.node.image_width, 3), dtype=np.uint8)

        # Mock Detection
        det_msg = Detection3DArray()
        det = Detection3D()
        det.bbox.center.position.x = 10.0
        det.bbox.center.position.y = 10.0
        det_msg.detections.append(det)

        # Mock Map
        map_gt_msg = MarkerArray()
        marker = Marker()
        marker.points.append(Point(x=5.0, y=5.0))
        marker.points.append(Point(x=5.0, y=15.0))
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # Green
        map_gt_msg.markers.append(marker)

        # Call drawing functions
        self.node.draw_map(bev_image, map_gt_msg)
        self.node.draw_detection_pred(bev_image, det_msg)

        # Verification
        # Check map pixel color (Green)
        map_px, map_py = self.node.world_to_pixel(5.0, 10.0)
        self.assertTrue(np.any(bev_image[map_py-1:map_py+2, map_px-1:map_px+2] == [0, 255, 0]))

        # Check detection color
        det_color = COLOR_MAPPING[1].tolist()
        det_px, det_py = self.node.world_to_pixel(10.0, 10.0)
        self.assertTrue(np.any(np.all(bev_image[det_py-5:det_py+5, det_px-5:det_px+5] == det_color, axis=-1)))

if __name__ == '__main__':
    unittest.main()