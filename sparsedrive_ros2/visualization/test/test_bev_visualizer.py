import unittest
import cv2
import numpy as np
from pyquaternion import Quaternion

from vision_msgs.msg import Detection3DArray, Detection3D
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Quaternion as GeoQuaternion
from multipath_msgs.msg import MultiPath

from ..drawing_utils import draw_bev

class TestBEVVisualizer(unittest.TestCase):

    def test_draw_bev(self):
        image_width, image_height, resolution = 800, 800, 0.1

        # Mock detections
        det_array = Detection3DArray()
        det = Detection3D()
        det.bbox.center.position.x = 5.0
        det.bbox.center.position.y = 2.0
        det.bbox.size.x = 4.0
        det.bbox.size.y = 2.0
        q = Quaternion(axis=[0, 0, 1], angle=np.pi / 4)
        det.bbox.center.orientation = GeoQuaternion(x=q.x, y=q.y, z=q.z, w=q.w)
        det_array.detections.append(det)

        # Mock trajectories
        multi_path = MultiPath()
        path = Path()
        for i in range(5):
            pose = PoseStamped()
            pose.pose.position.x = 1.0 * i
            pose.pose.position.y = 0.5 * i
            path.poses.append(pose)
        multi_path.paths.append(path)

        # Mock predicted plan
        plan_path = Path()
        for i in range(10):
            pose = PoseStamped()
            pose.pose.position.x = 1.2 * i
            pose.pose.position.y = -0.3 * i
            plan_path.poses.append(pose)

        # Mock ground truth plan
        plan_gt_path = Path()
        for i in range(10):
            pose = PoseStamped()
            pose.pose.position.x = 1.2 * i
            pose.pose.position.y = -0.4 * i
            plan_gt_path.poses.append(pose)

        # Mock map data
        map_data = np.zeros((image_height, image_width), dtype=np.uint8)
        map_data[100:200, 300:400] = 1
        map_pred_msg = Image(data=map_data.tobytes(), height=image_height, width=image_width, encoding="mono8")
        map_gt_msg = Image(data=map_data.tobytes(), height=image_height, width=image_width, encoding="mono8")


        bev_image = draw_bev(
            det_array, multi_path, plan_path,
            image_width, image_height, resolution,
            map_pred_msg, map_gt_msg, plan_gt_path
        )

        cv2.imwrite("test_bev_visualization.png", bev_image)

        self.assertIsNotNone(bev_image)
        self.assertEqual(bev_image.shape, (image_height, image_width, 3))
        self.assertTrue(np.any(bev_image > 0))

if __name__ == '__main__':
    unittest.main()