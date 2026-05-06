#!/usr/bin/env python3
"""
save_mei_undistort_maps.py  –  Compute MEI remap maps and save as binary float32 files.

libmei_undistort.so loads these at startup and uploads them to GPU once.
Output files: maps/cam_N_map_x.bin, maps/cam_N_map_y.bin

Usage:
  source ../venv/bin/activate
  python save_mei_undistort_maps.py
  python save_mei_undistort_maps.py --out-size 900 --fov 120
"""

import sys, os, argparse, re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CAMERAS   = ('02', '03')
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, '..', 'calibration')
MAPS_DIR  = os.path.join(BASE_DIR, 'maps')

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


def main():
    ap = argparse.ArgumentParser(
        description='Save MEI remap maps as binary float32 for nvds_undistort')
    ap.add_argument('--out-size', type=int,   default=1400)
    ap.add_argument('--fov',      type=float, default=120.0)
    args = ap.parse_args()

    os.makedirs(MAPS_DIR, exist_ok=True)

    for idx, cam_id in enumerate(CAMERAS):
        yaml = os.path.join(CALIB_DIR, f'image_{cam_id}.yaml')
        params = load_mei_params(yaml)
        map_x, map_y = build_undistort_maps(
            params, args.out_size, args.out_size, args.fov)

        mx_path = os.path.join(MAPS_DIR, f'cam_{idx}_mei_undistort_map_x.bin')
        my_path = os.path.join(MAPS_DIR, f'cam_{idx}_mei_undistort_map_y.bin')

        # Save as raw float32, C-contiguous row-major
        np.ascontiguousarray(map_x, dtype=np.float32).tofile(mx_path)
        np.ascontiguousarray(map_y, dtype=np.float32).tofile(my_path)

        print(f'cam {cam_id} (idx {idx}): {map_x.shape}  '
              f'→  {mx_path}  {my_path}')

    print(f'\nSaved to {MAPS_DIR}/')
    print(f'Use in config_undistort.txt:')

    for idx, cam_id in enumerate(CAMERAS):
        print(f'  cam-{idx}-map-x={MAPS_DIR}/cam_{idx}_mei_undistort_map_x.bin')
        print(f'  cam-{idx}-map-y={MAPS_DIR}/cam_{idx}_mei_undistort_map_y.bin')


if __name__ == '__main__':
    main()
