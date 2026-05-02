#!/usr/bin/env python3
"""
KITTI-360 Fisheye Undistortion  —  MEI (Unified Camera) Model

MEI forward projection  (3-D point P → pixel):
  1. Lift P onto unit sphere:     Ps = P / |P|
  2. Mirror shift:                m  = Ps[:2] / (Ps[2] + xi)
  3. Polynomial distortion        (k1, k2 radial;  p1, p2 tangential)
  4. Camera matrix                (gamma1, gamma2, u0, v0)

Undistortion builds a remap: for every output pixel we shoot a ray through
a virtual pinhole, apply the MEI forward model, and read back the colour
from the fisheye image.
"""

import re
import os
import argparse
import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────
# Parameter loading
# ──────────────────────────────────────────────────────────────

def load_mei_params(yaml_path: str) -> dict:
    """
    Load MEI parameters from an OpenCV-style YAML file.

    The file may start with '%YAML:1.0' which is not valid standard YAML,
    so we parse it with a regex rather than a YAML library.
    """
    with open(yaml_path, 'r') as f:
        content = f.read()

    def _float(key: str) -> float:
        m = re.search(
            rf'{re.escape(key)}:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
            content,
        )
        if m is None:
            raise ValueError(f"Key '{key}' not found in {yaml_path}")
        return float(m.group(1))

    def _int(key: str) -> int:
        m = re.search(rf'{re.escape(key)}:\s*(\d+)', content)
        if m is None:
            raise ValueError(f"Key '{key}' not found in {yaml_path}")
        return int(m.group(1))

    return {
        'xi':     _float('xi'),
        'k1':     _float('k1'),
        'k2':     _float('k2'),
        'p1':     _float('p1'),
        'p2':     _float('p2'),
        'gamma1': _float('gamma1'),
        'gamma2': _float('gamma2'),
        'u0':     _float('u0'),
        'v0':     _float('v0'),
        'width':  _int('image_width'),
        'height': _int('image_height'),
    }


# ──────────────────────────────────────────────────────────────
# Remap-map construction
# ──────────────────────────────────────────────────────────────

def build_undistort_maps(
    params: dict,
    out_w: int,
    out_h: int,
    fov_deg: float = 90.0,
) -> tuple:
    """
    Return (map_x, map_y) float32 arrays for cv2.remap.

    For each output pixel (u, v) we:
      1. Back-project through a virtual pinhole  →  3-D ray (X, Y, 1)
      2. Forward-project through the MEI model   →  source pixel in fisheye image
    """
    xi, k1, k2, p1, p2 = (
        params['xi'], params['k1'], params['k2'],
        params['p1'], params['p2'],
    )
    gamma1, gamma2, u0, v0 = (
        params['gamma1'], params['gamma2'],
        params['u0'],     params['v0'],
    )

    # Virtual pinhole focal length for the desired horizontal FOV
    f  = (out_w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    cx = out_w / 2.0
    cy = out_h / 2.0

    # Dense pixel grid of the output image
    u_grid, v_grid = np.meshgrid(
        np.arange(out_w, dtype=np.float64),
        np.arange(out_h, dtype=np.float64),
    )

    # ── Step 1: virtual pinhole → normalised 3-D ray ──────────
    X = (u_grid - cx) / f
    Y = (v_grid - cy) / f
    Z = np.ones_like(X)

    # ── Step 2: project onto unit sphere ──────────────────────
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    xs, ys, zs = X / norm, Y / norm, Z / norm

    # ── Step 3: MEI mirror shift ──────────────────────────────
    # (zs + xi) > 0 always when xi > 0 and Z = 1 > 0
    denom = zs + xi
    mx = xs / denom
    my = ys / denom

    # ── Step 4: polynomial distortion ─────────────────────────
    r2  = mx**2 + my**2
    r4  = r2 * r2
    rad = 1.0 + k1 * r2 + k2 * r4

    mx_d = mx * rad + 2.0 * p1 * mx * my      + p2 * (r2 + 2.0 * mx**2)
    my_d = my * rad + p1 * (r2 + 2.0 * my**2) + 2.0 * p2 * mx * my

    # ── Step 5: camera intrinsics → source pixel ──────────────
    map_x = (gamma1 * mx_d + u0).astype(np.float32)
    map_y = (gamma2 * my_d + v0).astype(np.float32)

    return map_x, map_y


# ──────────────────────────────────────────────────────────────
# Public undistortion helper
# ──────────────────────────────────────────────────────────────

def undistort_mei(
    img: np.ndarray,
    params: dict,
    out_w: int | None = None,
    out_h: int | None = None,
    fov_deg: float = 90.0,
) -> np.ndarray:
    """Undistort a MEI fisheye image to a rectilinear perspective image."""
    if out_w is None:
        out_w = params['width']
    if out_h is None:
        out_h = params['height']

    map_x, map_y = build_undistort_maps(params, out_w, out_h, fov_deg)
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Undistort KITTI-360 fisheye images using the MEI camera model.'
    )
    parser.add_argument('--fov',      type=float, default=90.0,
                        help='Output horizontal FOV in degrees (default: 90)')
    parser.add_argument('--out-size', type=int,   default=700,
                        help='Output image size in pixels (square, default: 700)')
    parser.add_argument('--base-dir', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory containing images and calibration/')
    args = parser.parse_args()

    base_dir  = args.base_dir
    calib_dir = os.path.join(base_dir, 'calibration')
    out_w = out_h = args.out_size
    fov   = args.fov

    cameras = [
        ('image_02', 'image_02.yaml', 'image_02.png'),
        ('image_03', 'image_03.yaml', 'image_03.png'),
    ]

    results = {}
    for name, yaml_file, img_file in cameras:
        yaml_path = os.path.join(calib_dir, yaml_file)
        img_path  = os.path.join(base_dir,  img_file)

        if not os.path.exists(yaml_path):
            print(f'[{name}] calibration file not found: {yaml_path}')
            continue
        if not os.path.exists(img_path):
            print(f'[{name}] image not found: {img_path}')
            continue

        params = load_mei_params(yaml_path)
        print(f'\n[{name}] MEI parameters')
        print(f'  xi     = {params["xi"]:.6f}  (mirror parameter)')
        print(f'  k1,k2  = {params["k1"]:.6f}, {params["k2"]:.6f}  (radial distortion)')
        print(f'  p1,p2  = {params["p1"]:.6f}, {params["p2"]:.6f}  (tangential distortion)')
        print(f'  focal  = ({params["gamma1"]:.2f}, {params["gamma2"]:.2f})')
        print(f'  centre = ({params["u0"]:.2f}, {params["v0"]:.2f})')

        img = cv2.imread(img_path)
        if img is None:
            print(f'  ERROR: cannot read {img_path}')
            continue
        print(f'  Input : {img.shape[1]}×{img.shape[0]}  →  Output: {out_w}×{out_h}  FOV={fov}°')

        undistorted = undistort_mei(img, params, out_w, out_h, fov)

        out_path = os.path.join(base_dir, f'undistort_{name}.png')
        cv2.imwrite(out_path, undistorted)
        print(f'  Saved → {out_path}')
        results[name] = (img, undistorted)

    # ── Comparison montage ─────────────────────────────────────
    if len(results) == 2:
        cols = []
        for name, (orig, undist) in results.items():
            orig_small = cv2.resize(orig, (out_w, out_h))

            def label_bar(text, w):
                bar = np.zeros((28, w, 3), dtype=np.uint8)
                cv2.putText(bar, text, (6, 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                            cv2.LINE_AA)
                return bar

            col = np.vstack([
                label_bar(f'{name} — original',    out_w), orig_small,
                label_bar(f'{name} — undistorted', out_w), undist,
            ])
            cols.append(col)

        montage_path = os.path.join(base_dir, 'undistort_comparison.png')
        cv2.imwrite(montage_path, np.hstack(cols))
        print(f'\nComparison montage → {montage_path}')


if __name__ == '__main__':
    main()
