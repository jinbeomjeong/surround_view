#!/usr/bin/env python3
"""
save_maps.py  –  Compute MEI remap maps and save as binary float32 files.

libmei_undistort.so loads these at startup and uploads them to GPU once.
Output files: maps/cam_N_map_x.bin, maps/cam_N_map_y.bin

Usage:
  source ../venv/bin/activate
  python save_maps.py                          # defaults: 700×700, FOV 90°
  python save_maps.py --out-size 900 --fov 120
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from undistort_fisheye import load_mei_params, build_undistort_maps

CAMERAS   = ('02', '03')
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, '..', 'calibration')
MAPS_DIR  = os.path.join(BASE_DIR, 'maps')


def main():
    ap = argparse.ArgumentParser(
        description='Save MEI remap maps as binary float32 for nvds_undistort')
    ap.add_argument('--out-size', type=int,   default=700)
    ap.add_argument('--fov',      type=float, default=90.0)
    args = ap.parse_args()

    os.makedirs(MAPS_DIR, exist_ok=True)

    for idx, cam_id in enumerate(CAMERAS):
        yaml = os.path.join(CALIB_DIR, f'image_{cam_id}.yaml')
        params = load_mei_params(yaml)
        map_x, map_y = build_undistort_maps(
            params, args.out_size, args.out_size, args.fov)

        mx_path = os.path.join(MAPS_DIR, f'cam_{idx}_map_x.bin')
        my_path = os.path.join(MAPS_DIR, f'cam_{idx}_map_y.bin')

        # Save as raw float32, C-contiguous row-major
        np.ascontiguousarray(map_x, dtype=np.float32).tofile(mx_path)
        np.ascontiguousarray(map_y, dtype=np.float32).tofile(my_path)

        print(f'cam {cam_id} (idx {idx}): {map_x.shape}  '
              f'→  {mx_path}  {my_path}')

    print(f'\nSaved to {MAPS_DIR}/')
    print(f'Use in config_undistort.txt:')
    for idx, cam_id in enumerate(CAMERAS):
        print(f'  cam-{idx}-map-x={MAPS_DIR}/cam_{idx}_map_x.bin')
        print(f'  cam-{idx}-map-y={MAPS_DIR}/cam_{idx}_map_y.bin')


if __name__ == '__main__':
    main()
