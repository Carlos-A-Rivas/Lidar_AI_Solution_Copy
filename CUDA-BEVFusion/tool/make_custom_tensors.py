# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Generate calibration tensors for a custom camera/LiDAR setup.

The values in this script come from the user's extrinsic and intrinsic
calibration.  Running it will produce ``.tensor`` files compatible with
``main.cpp`` or ``tool/pybev.py`` under ``custom-example/``.
"""

import os
from pathlib import Path
import numpy as np
from tensor import save

# Output directory
OUT_DIR = "custom-example"

# Extrinsics from the provided URDF (metres and radians)
# List of camera poses relative to ego (meters, radians)
CAMERA2EGO_TRANSLATIONS = [
    [2.232, 0.180, 0.442],
]
CAMERA2EGO_RPYS = [
    [0.0, 0.0, 0.0],
]

LIDAR2EGO_TRANSLATION = [2.222, 0.005, 0.448]
LIDAR2EGO_RPY = [0.0, 0.0, 0.0]

# Transform from lidar frame to camera frame
# I THINK THESE NUMBERS ARE FOR CAMERA2LIDAR (NOT LIDAR2CAMERA)
LIDAR2CAMERA_TRANSLATIONS = [
    [-5.55111512e-18, 0.394218218, 0.120797336],
]
LIDAR2CAMERA_RPYS = [
    [-1.5184364492350666, 0.0, -1.5707963267948966],
]

# Camera intrinsic matrix
CAMERA_MATRICES = [
    [
        [3563.6981563173754, 0.0, 1038.2496911952232],
        [0.0, 3574.349298227205, 147.84753651679904],
        [0.0, 0.0, 1.0],
    ],
]


def rpy_to_matrix(r, p, y):
    """Return 3x3 rotation matrix from roll/pitch/yaw (radians)."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)


def pose_to_matrix(x, y, z, r, p, yaw):
    """Create a 4x4 transform matrix from translation and RPY."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rpy_to_matrix(r, p, yaw)
    mat[:3, 3] = [x, y, z]
    return mat


os.makedirs(OUT_DIR, exist_ok=True)

camera2ego = np.stack([
    pose_to_matrix(*t, *r) for t, r in zip(CAMERA2EGO_TRANSLATIONS, CAMERA2EGO_RPYS)
])[None, ...]
lidar2ego = pose_to_matrix(*LIDAR2EGO_TRANSLATION, *LIDAR2EGO_RPY)[None]

lidar2camera = np.stack([
    pose_to_matrix(*t, *r) for t, r in zip(LIDAR2CAMERA_TRANSLATIONS, LIDAR2CAMERA_RPYS)
])[None, ...]
camera2lidar = np.linalg.inv(lidar2camera)

camera_intrinsics = np.stack([
    np.pad(np.array(m, dtype=np.float32), ((0,1),(0,1)), 'constant', constant_values=((0,0),(0,0)))
    for m in CAMERA_MATRICES
])[None, ...]

lidar2image = camera_intrinsics @ lidar2camera
img_aug_matrix = np.tile(np.eye(4, dtype=np.float32), (1, len(CAMERA_MATRICES), 1, 1))
lidar_aug_matrix = np.tile(np.eye(4, dtype=np.float32), (1, 1, 1, 1))


def save_tensor(name, array):
    save(array.astype(np.float32), os.path.join(OUT_DIR, f"{name}.tensor"))


save_tensor("camera2ego", camera2ego)
save_tensor("lidar2ego", lidar2ego)
save_tensor("camera2lidar", camera2lidar)
save_tensor("lidar2camera", lidar2camera)
save_tensor("camera_intrinsics", camera_intrinsics)
save_tensor("lidar2image", lidar2image)
save_tensor("img_aug_matrix", img_aug_matrix)
save_tensor("lidar_aug_matrix", lidar_aug_matrix)

print(f"Tensors written to {OUT_DIR}")
