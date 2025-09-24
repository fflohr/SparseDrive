import unittest
import numpy as np
import cv2
from sparsedrive_ros2.visualization.bev_render import BEVRendererNode, COLOR_MAPPING
from vision_msgs.msg import Detection3DArray, Detection3D
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
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
        det.bbox.size.x = 4.0
        det.bbox.size.y = 2.0
        det_msg.detections.append(det)

        # Mock Map
        map_data = np.zeros((200, 200), dtype=np.int8)
        map_data[100:101, 100:101] = 100 # A single occupied cell
        map_gt_msg = OccupancyGrid()
        map_gt_msg.info.width = 200
        map_gt_msg.info.height = 200
        map_gt_msg.info.resolution = 0.5
        map_gt_msg.info.origin.position.x = -50.0
        map_gt_msg.info.origin.position.y = -50.0
        map_gt_msg.data = map_data.flatten().tolist()

        # Call drawing functions
        self.node.draw_map(bev_image, map_gt_msg, (0, 255, 0)) # Green
        self.node.draw_detection_pred(bev_image, det_msg)

        # Verification
        # Check map pixel color
        map_world_x = -50.0 + 100 * 0.5
        map_world_y = -50.0 + 100 * 0.5
        map_px, map_py = self.node.world_to_pixel(map_world_x, map_world_y)
        self.assertTrue(np.all(bev_image[map_py, map_px] == [0, 255, 0]))

        # Check detection color
        det_color = COLOR_MAPPING[1].tolist() # First detection uses the second color
        det_px, det_py = self.node.world_to_pixel(10.0, 10.0)
        self.assertTrue(np.any(np.all(bev_image[det_py-5:det_py+5, det_px-5:det_px+5] == det_color, axis=-1)))

if __name__ == '__main__':
    unittest.main()