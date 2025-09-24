import os
import numpy as np
import torch
from pyquaternion import Quaternion
from PIL import Image

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmcv.parallel import DataContainer as DC
from ament_index_python.packages import get_package_share_directory

from .static_transforms import CAM_FRONT_RIGHT, LIDAR_TOP

package_share = get_package_share_directory('sparsedrive_ros2')

CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
)
MAP_CLASSES = (
        'ped_crossing',
        'divider',
        'boundary',
)

# img_norm_cfg = dict(
#         mean=[123.675, 116.28, 103.53],  # ImageNet mean
#         std=[58.395, 57.12, 57.375],     # ImageNet std
#         to_rgb=True,
#     )

img_norm_cfg = dict(
        mean=[103.53, 116.28, 123.675],  # ImageNet mean
        std=[ 57.375, 57.12, 58.395],     # ImageNet std
        to_rgb=False,
    )

test_pipeline = [
        # dict(type="ResizeCropFlipImage"),
        dict(type="NormalizeMultiviewImage", **img_norm_cfg),
        dict(type="NuScenesSparse4DAdaptor"),
    ]

keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            'ego_status',
            'gt_ego_fut_cmd',
        ]
meta_keys=["T_global", "T_global_inv", "timestamp"]


# Transforms in nuScenes format
def format_transforms():
    cam_intrinsic = np.array(CAM_FRONT_RIGHT['camera_intrinsic']).astype(np.float32)

    l2e_t = np.array(LIDAR_TOP['translation'], dtype=np.float32)
    l2e_r = np.array(LIDAR_TOP['rotation'], dtype=np.float32)

    l2e_r_mat = Quaternion(l2e_r).rotation_matrix

    c2e_t = np.array(CAM_FRONT_RIGHT['translation'], dtype=np.float32)
    c2e_r = np.array(CAM_FRONT_RIGHT['rotation'], dtype=np.float32)

    c2e_r_mat = Quaternion(c2e_r).rotation_matrix

    lidar2ego_mat = np.eye(4)
    lidar2ego_mat[:3, :3] = l2e_r_mat
    lidar2ego_mat[:3, 3] = l2e_t

    cam2ego_mat = np.eye(4)
    cam2ego_mat[:3, :3] = c2e_r_mat
    cam2ego_mat[:3, 3] = c2e_t

    cam2lidar_mat = np.linalg.inv(np.linalg.inv(cam2ego_mat) @ lidar2ego_mat)

    cam2lidar_rotation = cam2lidar_mat[:3, :3]
    cam2lidar_translation = cam2lidar_mat[:3, 3]

    lidar2cam_r = np.linalg.inv(cam2lidar_rotation)
    lidar2cam_t = (
        cam2lidar_translation @ lidar2cam_r.T
    )
    lidar2cam_mat = np.eye(4)
    lidar2cam_mat[:3, :3] = lidar2cam_r.T
    lidar2cam_mat[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: cam_intrinsic.shape[0], : cam_intrinsic.shape[1]] = cam_intrinsic
    lidar2img_mat = viewpad @ lidar2cam_mat.T

    return lidar2ego_mat, lidar2cam_mat, lidar2img_mat



def get_augmentation(data_aug_conf, test_mode=True):
    if data_aug_conf is None:
        return None
    H, W = data_aug_conf["H"], data_aug_conf["W"]
    fH, fW = data_aug_conf["final_dim"]
    if not test_mode:
        resize = np.random.uniform(*data_aug_conf["resize_lim"])
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = (
            int(
                (1 - np.random.uniform(*data_aug_conf["bot_pct_lim"]))
                * newH
            )
            - fH
        )
        crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        if data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
            flip = True
        rotate = np.random.uniform(*data_aug_conf["rot_lim"])
        rotate_3d = np.random.uniform(*data_aug_conf["rot3d_range"])
    else:
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = (
            int((1 - np.mean(data_aug_conf["bot_pct_lim"])) * newH)
            - fH
        )
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        rotate_3d = 0
    aug_config = {
        "resize": resize,
        "resize_dims": resize_dims,
        "crop": crop,
        "flip": flip,
        "rotate": rotate,
        "rotate_3d": rotate_3d,
    }
    return aug_config

def get_model(config, device):
    # Load config and model
    cfg = Config.fromfile(os.path.join(package_share, "config", "sparsedrive_small_stage2.py"))  # replace with your config
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, "/home/kanak/SparseDrive/ckpt/sparsedrive_stage2.pth", map_location="cpu")

    if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = CLASSES

    model.eval().to(device)
    return model

def init_data(imgH=1200, imgW=2200):
    #  dict_keys(['img_filename', 'cam_intrinsic', 'img', 'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'lidar2cam', 'lidar2img', 'lidar2global', 'timestamp', 'ego_status', 'gt_ego_fut_cmd'])

    results = {}

    N = 6
    input_shape = (704, 384) #(704, 256)
    data_aug_conf = {
        "resize_lim": (0.40, 0.47),
        "final_dim": input_shape[::-1],
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (-5.4, 5.4),
        "H": imgH,
        "W": imgW,
        "rand_flip": True,
        "rot3d_range": [0, 0],
    }
    cam_intrinsic = [np.eye(3) for _ in range(N)]
    # imgs = np.stack([np.zeros((imgH, imgW, 3)) for _ in range(N)], dtype=np.float32, axis=-1)
    imgs = np.stack([np.zeros((input_shape[1], input_shape[0], 3)) for _ in range(N)], dtype=np.float32, axis=-1)
    lidar2cam = np.stack([np.eye(4) for _ in range(N)])
    lidar2img = np.stack([np.eye(4) for _ in range(N)])
    lidar2global = np.eye(4)
    timestamp = 0.0
    command = np.array([0, 0, 1]).astype(np.float32)  # Go Straight
    ego_status = [0] * 10
    ego_status= np.array(ego_status).astype(np.float32)
    results["img"] = [imgs[..., i] for i in range(imgs.shape[-1])]
    results["img_shape"] = imgs.shape
    results["ori_shape"] = imgs.shape
    results["pad_shape"] = imgs.shape
    results["scale_factor"] = 1.0
    results["img_norm_cfg"] = img_norm_cfg
    results["cam_intrinsic"] = cam_intrinsic
    results['lidar2cam'] = np.array(lidar2cam)
    results['lidar2img'] = np.array(lidar2img)
    results['lidar2global'] = np.array(lidar2global)
    results['timestamp'] = np.array(timestamp)
    results['ego_status'] = ego_status
    results['gt_ego_fut_cmd'] = command

    # additional
    results['data_aug_conf'] = data_aug_conf
    results['aug_config'] = get_augmentation(results['data_aug_conf'])
    return results

def get_data(results, img, intrinsic, ego2global, lidar2ego, lidar2cam, lidar2img, timestamp):
    lidar2global = ego2global @ lidar2ego

    results['img'][0] = img
    results['lidar2global'] = lidar2global
    results['lidar2cam'][0] = lidar2cam
    results['lidar2img'][0] = lidar2img
    results['timestamp'] = np.array([timestamp])
    results['cam_intrinsic'][0] = intrinsic
    # results['aug_config'] = get_augmentation(results['data_aug_conf'])
    return results

def format_data(info):
    data = {}
    img_meta = {}
    for key in meta_keys:
        img_meta[key] = info[key]
    data['img_metas'] = [img_meta]
    for key in keys:
        if isinstance(info[key], DC):
            data[key] = info[key].data
        else:
            data[key] = info[key]
    

    if 'img' in data.keys():
        imgs = data['img'].unsqueeze(0).cuda()

    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            data[key] = torch.from_numpy(np.expand_dims(data[key], axis=0)).cuda()
        elif isinstance(data[key], DC):
            data[key] = data[key].data
        elif isinstance(data[key], float):
            data[key] = torch.Tensor(np.array(data[key]))

    if data['timestamp'].dim() == 2:
        data['timestamp'] = data['timestamp'].squeeze(0)

    if 'img' in data.keys():
            data.pop('img')

    return imgs, data

class ResizeCropFlipImage():
    def __call__(self, results):
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img, mat = self._img_transform(
                np.uint8(imgs[i]), aug_config,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= aug_config["resize"]
                # results["cam_intrinsic"][i][:3, :3] = (
                #     mat[:3, :3] @ results["cam_intrinsic"][i][:3, :3]
                # )

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _img_transform(self, img, aug_configs):
        # H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = aug_configs.get("resize_dims")
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)
        already_resized = aug_configs.get("already_resized", False)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        if not already_resized:
            img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix