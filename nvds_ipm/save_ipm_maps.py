#!/usr/bin/env python3
"""
Generate IPM remap maps for the 1400x1400 DeepStream IPM stage.

The maps are raw float32 arrays loaded by nvds_ipm/config_ipm_bev.txt.
The BEV image is not rotated here; bev.py
handles per-camera rotation later with nvvideoconvert flip-method.
"""

import os

import numpy as np


OUT_W, OUT_H = 1400, 1400
IMG_W, IMG_H = 1400, 1400
FOV_DEG = 130.0
CAM_HEIGHT_M = 1.65
PITCH_DEG = 0.0
ROLL_DEG = 0.0

# 20 m forward and 20 m lateral span. At 0.025 m/px this creates an
# 800x800 BEV region centered inside the 1400x1400 output canvas.
X_RANGE_M = (0.0, 20.0)
Y_RANGE_M = (-10.0, 10.0)
RES_M_PER_PX = 0.025

# The downstream pipeline rotates cam02/cam03 with nvvideoconvert flip-method,
# so the IPM maps themselves do not apply per-camera yaw rotation.
CAMERA_CONFIGS = (
    (0, '02', 0.0),
    (1, '03', 0.0),
)


def build_K(fov_deg: float, width: int, height: int) -> np.ndarray:
    f = (width / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    return np.array(
        [[f, 0.0, width / 2.0],
         [0.0, f, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, c, -s],
         [0.0, s, c]],
        dtype=np.float64,
    )


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [[c, 0.0, s],
         [0.0, 1.0, 0.0],
         [-s, 0.0, c]],
        dtype=np.float64,
    )


def rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [[c, -s, 0.0],
         [s, c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def build_R_world_to_cam(yaw: float, pitch: float, roll: float) -> np.ndarray:
    # World: X forward, Y left, Z up. Camera: x right, y down, z forward.
    # The base matrix maps a forward-facing ground camera into pinhole camera
    # coordinates; yaw then turns that camera left/right around world Z.
    base_world_to_cam = np.array(
        [[0.0, -1.0, 0.0],
         [0.0, 0.0, -1.0],
         [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    return base_world_to_cam @ rot_z(-yaw) @ rot_y(pitch) @ rot_x(roll)


def generate_ipm_map(
    out_w: int = OUT_W,
    out_h: int = OUT_H,
    img_w: int = IMG_W,
    img_h: int = IMG_H,
    fov: float = FOV_DEG,
    cam_height: float = CAM_HEIGHT_M,
    pitch: float = PITCH_DEG,
    yaw: float = 0.0,
    roll: float = ROLL_DEG,
    x_range: tuple[float, float] = X_RANGE_M,
    y_range: tuple[float, float] = Y_RANGE_M,
    res: float = RES_M_PER_PX,
) -> tuple[np.ndarray, np.ndarray]:
    K = build_K(fov, img_w, img_h)
    R = build_R_world_to_cam(yaw, pitch, roll)
    t = -R @ np.array([0.0, 0.0, cam_height], dtype=np.float64)

    bev_w = int(round((y_range[1] - y_range[0]) / res))
    bev_h = int(round((x_range[1] - x_range[0]) / res))
    if bev_w > out_w or bev_h > out_h:
        raise ValueError(
            f'Output canvas {out_w}x{out_h} is too small for BEV '
            f'{bev_w}x{bev_h}')

    u_bev = np.arange(bev_w, dtype=np.float64)
    v_bev = np.arange(bev_h, dtype=np.float64)
    uu_bev, vv_bev = np.meshgrid(u_bev, v_bev)

    Xw = x_range[1] - vv_bev * res
    Yw = y_range[1] - uu_bev * res
    Zw = np.zeros_like(Xw)

    Xc = R[0, 0] * Xw + R[0, 1] * Yw + R[0, 2] * Zw + t[0]
    Yc = R[1, 0] * Xw + R[1, 1] * Yw + R[1, 2] * Zw + t[1]
    Zc = R[2, 0] * Xw + R[2, 1] * Yw + R[2, 2] * Zw + t[2]

    valid = Zc > 1e-3
    u_img = np.where(valid, K[0, 0] * Xc / Zc + K[0, 2], -1.0)
    v_img = np.where(valid, K[1, 1] * Yc / Zc + K[1, 2], -1.0)

    map_x = np.full((out_h, out_w), -1.0, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1.0, dtype=np.float32)

    offset_x = (out_w - bev_w) // 2
    offset_y = (out_h - bev_h) // 2
    dst_y = slice(offset_y, offset_y + bev_h)
    dst_x = slice(offset_x, offset_x + bev_w)
    map_x[dst_y, dst_x] = u_img.astype(np.float32)
    map_y[dst_y, dst_x] = v_img.astype(np.float32)
    return map_x, map_y


def save_maps() -> None:
    maps_dir = os.path.join(os.path.dirname(__file__), 'maps')
    os.makedirs(maps_dir, exist_ok=True)

    for idx, cam_id, yaw in CAMERA_CONFIGS:
        map_x, map_y = generate_ipm_map(yaw=yaw)
        x_path = os.path.join(maps_dir, f'cam_{idx}_ipm_bev_x.bin')
        y_path = os.path.join(maps_dir, f'cam_{idx}_ipm_bev_y.bin')
        np.ascontiguousarray(map_x, dtype=np.float32).tofile(x_path)
        np.ascontiguousarray(map_y, dtype=np.float32).tofile(y_path)
        valid = (
            (map_x >= 0.0) & (map_x < IMG_W) &
            (map_y >= 0.0) & (map_y < IMG_H)
        )
        print(
            f'cam{cam_id} -> cam_{idx}: yaw={yaw:+.1f} deg, '
            f'shape={map_x.shape}, valid={int(valid.sum())}/{valid.size}'
        )
        print(f'  {x_path}')
        print(f'  {y_path}')

    print('IPM maps generated successfully in nvds_ipm/maps/')


if __name__ == '__main__':
    save_maps()
