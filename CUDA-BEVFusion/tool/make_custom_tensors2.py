# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Generate calibration tensors for a custom camera/LiDAR setup.

The values in this script come from the user's extrinsic and intrinsic
calibration.  Running it will produce `.tensor` files compatible with
`main.cpp` or `tool/pybev.py` under `custom-example/`.
"""

import os
import numpy as np
from tensor import save

# Output directory
OUT_DIR = "custom-example"

# Extrinsics from the provided URDF (meters and radians)
CAMERA2EGO_TRANSLATIONS = [
    [2.232, 0.180, 0.442],
]
CAMERA2EGO_RPYS = [
    [0.0, 0.0, 0.0],
]

LIDAR2EGO_TRANSLATION = [2.222, 0.005, 0.448]
LIDAR2EGO_RPY = [0.0, 0.0, 0.0]

# Camera intrinsic matrices (3x3)
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

# Make sure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Build sensor→ego transforms
camera2ego = np.stack([
    pose_to_matrix(*t, *r) for t, r in zip(CAMERA2EGO_TRANSLATIONS, CAMERA2EGO_RPYS)
])[None, ...]    # shape [1, n_cams, 4,4]
lidar2ego  = pose_to_matrix(*LIDAR2EGO_TRANSLATION, *LIDAR2EGO_RPY)[None, ...]  # shape [1, 4,4]

# Derive lidar→camera and camera→lidar from the two ego extrinsics
cam2ego = camera2ego[0, 0]    # 4×4
ego2cam = np.linalg.inv(cam2ego)
lidar2ego_mat = lidar2ego[0]  # 4×4
# true lidar→camera
lidar2camera_mat = ego2cam @ lidar2ego_mat
lidar2camera = lidar2camera_mat[None, None, ...]  # shape [1,1,4,4]
# inverse
camera2lidar = np.linalg.inv(lidar2camera_mat)[None, None, ...]

# Build correct 4×4 intrinsics (homogeneous)
camera_intrinsics = np.stack([np.eye(4, dtype=np.float32) for _ in CAMERA_MATRICES])[None, ...]
for i, K3 in enumerate(CAMERA_MATRICES):
    camera_intrinsics[0, i, :3, :3] = np.array(K3, dtype=np.float32)

# Projection: lidar points into each camera image
lidar2image = camera_intrinsics @ lidar2camera    # shape [1,n_cam,4,4]

# No augmentations (identity)
img_aug_matrix   = np.tile(np.eye(4, dtype=np.float32), (1, len(CAMERA_MATRICES), 1, 1))
lidar_aug_matrix = np.tile(np.eye(4, dtype=np.float32), (1, 1, 1, 1))

# Helper to write out .tensor files

def save_tensor(name, array):
    path = os.path.join(OUT_DIR, f"{name}.tensor")
    save(array.astype(np.float32), path)

# Save everything
save_tensor("camera2ego", camera2ego)
save_tensor("lidar2ego", lidar2ego)
save_tensor("lidar2camera", lidar2camera)
save_tensor("camera2lidar", camera2lidar)
save_tensor("camera_intrinsics", camera_intrinsics)
save_tensor("lidar2image", lidar2image)
save_tensor("img_aug_matrix", img_aug_matrix)
save_tensor("lidar_aug_matrix", lidar_aug_matrix)

print(f"Tensors written to {OUT_DIR}")
