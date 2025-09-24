import cv2
import numpy as np
from pyquaternion import Quaternion

def world_to_pixel(x, y, image_width, image_height, resolution):
    pixel_x = int(image_width / 2 - y / resolution)
    pixel_y = int(image_height / 2 - x / resolution)
    return pixel_x, pixel_y

def draw_bev(
    detection_msg, traj_msg, plan_msg,
    image_width, image_height, resolution,
    map_pred_msg=None, map_gt_msg=None, plan_gt_msg=None
):
    bev_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Draw Map GT
    if map_gt_msg:
        map_gt = np.frombuffer(map_gt_msg.data, dtype=np.uint8).reshape(map_gt_msg.height, map_gt_msg.width)
        map_gt_color = np.zeros_like(bev_image)
        map_gt_color[map_gt == 1] = [0, 255, 255]  # Cyan for GT map
        bev_image = cv2.addWeighted(bev_image, 1, map_gt_color, 0.5, 0)

    # Draw Map Pred
    if map_pred_msg:
        map_pred = np.frombuffer(map_pred_msg.data, dtype=np.uint8).reshape(map_pred_msg.height, map_pred_msg.width)
        map_pred_color = np.zeros_like(bev_image)
        map_pred_color[map_pred == 1] = [255, 255, 0]  # Yellow for predicted map
        bev_image = cv2.addWeighted(bev_image, 1, map_pred_color, 0.5, 0)

    # Draw Detections
    for detection in detection_msg.detections:
        center_x = detection.bbox.center.position.x
        center_y = detection.bbox.center.position.y
        q = Quaternion(
            w=detection.bbox.center.orientation.w,
            x=detection.bbox.center.orientation.x,
            y=detection.bbox.center.orientation.y,
            z=detection.bbox.center.orientation.z
        )
        yaw = q.yaw_pitch_roll[0]
        width = detection.bbox.size.x / resolution
        length = detection.bbox.size.y / resolution

        box_center_px = world_to_pixel(center_x, center_y, image_width, image_height, resolution)

        box = cv2.boxPoints(((box_center_px[0], box_center_px[1]), (width, length), -np.degrees(yaw)))
        box = np.int0(box)
        cv2.drawContours(bev_image, [box], 0, (0, 255, 0), 2)

    # Draw Trajectories
    for path in traj_msg.paths:
        points = []
        for pose in path.poses:
            points.append(world_to_pixel(pose.pose.position.x, pose.pose.position.y, image_width, image_height, resolution))
        if points:
            cv2.polylines(bev_image, [np.array(points, dtype=np.int32)], False, (255, 0, 0), 2)

    # Draw Plan GT
    if plan_gt_msg:
        plan_gt_points = []
        for pose in plan_gt_msg.poses:
            plan_gt_points.append(world_to_pixel(pose.pose.position.x, pose.pose.position.y, image_width, image_height, resolution))
        if plan_gt_points:
            cv2.polylines(bev_image, [np.array(plan_gt_points, dtype=np.int32)], False, (0, 255, 255), 2) # Cyan for GT plan

    # Draw Plan Pred
    plan_points = []
    for pose in plan_msg.poses:
        plan_points.append(world_to_pixel(pose.pose.position.x, pose.pose.position.y, image_width, image_height, resolution))
    if plan_points:
        cv2.polylines(bev_image, [np.array(plan_points, dtype=np.int32)], False, (0, 0, 255), 2)

    return bev_image