#!/usr/bin/env python3
"""
ipm.py  —  Inverse Perspective Mapping (bird's-eye view) for undistorted images.

The input image is assumed to have been produced by undistort_fisheye.py with a
virtual pinhole camera.  The virtual intrinsics are reconstructed from
--fov / --img-w / --img-h so they must match what was used during undistortion.

Coordinate conventions
----------------------
  World frame   : X forward, Y left, Z up  (right-handed vehicle body frame)
  Camera frame  : X right,   Y down, Z forward  (OpenCV standard)

Camera is mounted at height h above the flat ground plane (Z_world = 0).
Orientation is specified by (yaw, pitch, roll) applied in that order around
world Z → world Y → world X axes before aligning to camera frame.

Usage
-----
  python ipm.py
  python ipm.py --input undistort_image_02.png --cam-height 1.65 --cam-pitch 10
  python ipm.py --cam-yaw 90 --x-range 0 20 --y-range -15 15 --res 0.04
  python ipm.py --grid               # overlay metric grid on BEV
"""

import argparse
import os
import sys

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────
# Camera intrinsics
# ──────────────────────────────────────────────────────────────

def build_K(fov_deg: float, img_w: int, img_h: int) -> np.ndarray:
    """Virtual pinhole K from horizontal FOV and image size."""
    f = (img_w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    return np.array([[f,   0,   img_w / 2.0],
                     [0,   f,   img_h / 2.0],
                     [0,   0,   1.0        ]], dtype=np.float64)


# ──────────────────────────────────────────────────────────────
# Camera extrinsics
# ──────────────────────────────────────────────────────────────

def _Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float64)

def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)

def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


def build_R_world_to_cam(yaw_deg: float,
                          pitch_deg: float,
                          roll_deg: float) -> np.ndarray:
    """
    Rotation matrix  R  such that  P_cam = R @ P_world + t.

    Base alignment (yaw=pitch=roll=0, camera facing forward):
      cam X (right)   ←→  world -Y (right)
      cam Y (down)    ←→  world -Z (down)
      cam Z (forward) ←→  world  X (forward)

    Then yaw (world Z), pitch (world Y after yaw), roll (world X after yaw+pitch)
    are applied to rotate the camera's field of view.
    """
    # Base: cam axes expressed in world frame (columns = cam axes in world coords)
    #   cam_Z (forward) = world_X  →  R_base row 2 = [1, 0, 0]
    #   cam_X (right)   = -world_Y →  R_base row 0 = [0,-1, 0]
    #   cam_Y (down)    = -world_Z →  R_base row 1 = [0, 0,-1]
    R_base = np.array([[ 0, -1,  0],
                        [ 0,  0, -1],
                        [ 1,  0,  0]], dtype=np.float64)

    # Orientation offset in world frame: yaw → pitch → roll
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)   # positive = nose down
    r = np.radians(roll_deg)    # positive = right side down

    # R_offset rotates world frame before applying R_base
    # yaw around world Z, then pitch around world Y, then roll around world X
    R_offset = _Rx(r) @ _Ry(-p) @ _Rz(y)

    return R_base @ R_offset.T  # world-to-camera


# ──────────────────────────────────────────────────────────────
# IPM core
# ──────────────────────────────────────────────────────────────

def ipm(img: np.ndarray,
        K: np.ndarray,
        cam_height: float,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        x_range: tuple,
        y_range: tuple,
        res: float) -> np.ndarray:
    """
    Produce a bird's-eye-view image via inverse perspective mapping.

    Parameters
    ----------
    img        : undistorted source image  (H, W [, C])
    K          : 3×3 virtual pinhole intrinsic matrix
    cam_height : camera height above ground plane in metres
    yaw_deg    : camera yaw offset in degrees (0 = forward, +90 = left)
    pitch_deg  : camera downward tilt in degrees (positive = looking down)
    roll_deg   : camera roll in degrees (positive = right side down)
    x_range    : (x_near, x_far) metres in front of the camera
    y_range    : (y_right, y_left) metres laterally
                 (negative = right side, positive = left side)
    res        : metres per BEV pixel

    Returns
    -------
    bev : bird's-eye-view image, same dtype as img
    """
    R = build_R_world_to_cam(yaw_deg, pitch_deg, roll_deg)

    # Camera position in world: (0, 0, h)  →  translation in camera frame
    t = -R @ np.array([0.0, 0.0, cam_height])

    # BEV output grid
    bev_w = int(round((y_range[1] - y_range[0]) / res))
    bev_h = int(round((x_range[1] - x_range[0]) / res))

    u_bev = np.arange(bev_w, dtype=np.float64)
    v_bev = np.arange(bev_h, dtype=np.float64)
    uu, vv = np.meshgrid(u_bev, v_bev)

    # World coordinates  (ground plane: Z_world = 0)
    # BEV top    → x_far,  BEV bottom → x_near
    # BEV left   → y_left, BEV right  → y_right
    Xw = x_range[1] - vv * res
    Yw = y_range[1] - uu * res
    Zw = np.zeros_like(Xw)

    # Transform to camera frame
    Xc = R[0, 0] * Xw + R[0, 1] * Yw + R[0, 2] * Zw + t[0]
    Yc = R[1, 0] * Xw + R[1, 1] * Yw + R[1, 2] * Zw + t[1]
    Zc = R[2, 0] * Xw + R[2, 1] * Yw + R[2, 2] * Zw + t[2]

    # Perspective projection
    valid = Zc > 1e-3
    u_img = np.where(valid, K[0, 0] * Xc / Zc + K[0, 2], -1.0)
    v_img = np.where(valid, K[1, 1] * Yc / Zc + K[1, 2], -1.0)

    map_x = u_img.astype(np.float32)
    map_y = v_img.astype(np.float32)

    bev = cv2.remap(img, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
    return bev


# ──────────────────────────────────────────────────────────────
# BEV rotation  (for around-view alignment)
# ──────────────────────────────────────────────────────────────

def rotate_bev(bev: np.ndarray,
               rotate_deg: float,
               origin_col: int,
               origin_row: int) -> np.ndarray:
    """
    Rotate the BEV image by rotate_deg degrees (clockwise positive) around
    the camera origin pixel (origin_col, origin_row).

    The output canvas is large enough that no content inside a circle of
    radius = max(bev diagonal / 2) is clipped.  The origin pixel is kept
    at the same position in the output so that multiple rotated BEVs from
    different cameras can be alpha-blended on a shared canvas.
    """
    h, w = bev.shape[:2]

    # Expand canvas so rotation around origin does not clip corners.
    # Required half-size = distance from origin to the farthest corner.
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)
    origin  = np.array([origin_col, origin_row], dtype=np.float64)
    radius  = int(np.ceil(np.max(np.linalg.norm(corners - origin, axis=1))))

    canvas_w = 2 * radius
    canvas_h = 2 * radius
    pad_x = radius - origin_col
    pad_y = radius - origin_row

    # Place original BEV onto the expanded canvas
    canvas = np.zeros((canvas_h, canvas_w, bev.shape[2] if bev.ndim == 3 else 1),
                      dtype=bev.dtype)
    if bev.ndim == 2:
        canvas = np.zeros((canvas_h, canvas_w), dtype=bev.dtype)

    y0, y1 = pad_y, pad_y + h
    x0, x1 = pad_x, pad_x + w
    canvas[y0:y1, x0:x1] = bev

    # Rotate around the canvas centre (= camera origin after padding)
    cx, cy = radius, radius
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), -rotate_deg, 1.0)
    rotated = cv2.warpAffine(canvas, M, (canvas_w, canvas_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    return rotated


# ──────────────────────────────────────────────────────────────
# Black-background crop & scale
# ──────────────────────────────────────────────────────────────

def crop_black(img: np.ndarray) -> np.ndarray:
    """Crop away the surrounding black border (all-zero rows/columns)."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img  # fully black — return as-is
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img[r0:r1 + 1, c0:c1 + 1]


# ──────────────────────────────────────────────────────────────
# Grid overlay
# ──────────────────────────────────────────────────────────────

def draw_grid(bev: np.ndarray,
              x_range: tuple,
              y_range: tuple,
              res: float,
              interval_m: float = 5.0) -> np.ndarray:
    """Draw a metric grid on top of a BEV image."""
    out = bev.copy()
    bev_h, bev_w = out.shape[:2]
    colour = (60, 60, 60)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    def x_to_row(x):
        return int(round((x_range[1] - x) / res))

    def y_to_col(y):
        return int(round((y_range[1] - y) / res))

    # Horizontal lines (constant X / forward distance)
    x = np.arange(np.ceil(x_range[0] / interval_m) * interval_m,
                  x_range[1] + 1e-6, interval_m)
    for xi in x:
        row = x_to_row(xi)
        if 0 <= row < bev_h:
            cv2.line(out, (0, row), (bev_w - 1, row), colour, 1)
            cv2.putText(out, f'{xi:.0f}m', (4, row - 3),
                        font, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

    # Vertical lines (constant Y / lateral)
    y = np.arange(np.ceil(y_range[0] / interval_m) * interval_m,
                  y_range[1] + 1e-6, interval_m)
    for yi in y:
        col = y_to_col(yi)
        if 0 <= col < bev_w:
            cv2.line(out, (col, 0), (col, bev_h - 1), colour, 1)
            label = f'{yi:+.0f}m'
            cv2.putText(out, label, (col + 3, bev_h - 6),
                        font, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

    # Camera origin marker
    origin_row = x_to_row(0.0)
    origin_col = y_to_col(0.0)
    if 0 <= origin_row < bev_h and 0 <= origin_col < bev_w:
        cv2.drawMarker(out, (origin_col, origin_row), (0, 200, 255),
                       cv2.MARKER_CROSS, 20, 2)

    return out


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(
        description='Inverse Perspective Mapping — undistorted image → BEV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output
    ap.add_argument('--input',  default=os.path.join(base_dir, 'undistort_image_02.png'),
                    help='Path to undistorted input image')
    ap.add_argument('--output', default=None,
                    help='Output BEV path (default: <stem>_bev.png)')

    # Virtual pinhole parameters  (must match undistort_fisheye.py settings)
    ap.add_argument('--fov',   type=float, default=130.0,
                    help='Horizontal FOV used during undistortion (degrees)')
    ap.add_argument('--img-w', type=int,   default=700,
                    help='Undistorted image width (pixels)')
    ap.add_argument('--img-h', type=int,   default=700,
                    help='Undistorted image height (pixels)')

    # Camera extrinsics
    ap.add_argument('--cam-height', type=float, default=1.65,
                    help='Camera height above ground (metres)')
    ap.add_argument('--cam-pitch',  type=float, default=10.0,
                    help='Camera downward tilt (degrees, positive = looking down)')
    ap.add_argument('--cam-roll',   type=float, default=0.0,
                    help='Camera roll (degrees, positive = right side down)')
    ap.add_argument('--cam-yaw',    type=float, default=0.0,
                    help='Camera yaw offset from vehicle forward (degrees, '
                         'positive = left;  use 90 for left-facing camera)')

    # BEV output area
    ap.add_argument('--x-range', type=float, nargs=2, default=[0.0, 20.0],
                    metavar=('NEAR', 'FAR'),
                    help='Forward range in metres')
    ap.add_argument('--y-range', type=float, nargs=2, default=[-10.0, 10.0],
                    metavar=('RIGHT', 'LEFT'),
                    help='Lateral range in metres (negative=right, positive=left)')
    ap.add_argument('--res',     type=float, default=0.05,
                    help='BEV resolution (metres per pixel)')

    # Extras
    ap.add_argument('--trim-top', type=int, default=250,
                    help='Pixels to remove from the top of the BEV image '
                         '(top = far distance; use this to cut low-quality area). '
                         f'At default --res 0.05 m/px: 1 px ≈ 0.05 m.')
    ap.add_argument('--rotate', type=float, default=0.0,
                    help='Clockwise rotation of the BEV image in degrees, applied '
                         'around the camera origin (vehicle position in BEV). '
                         'Aligns each camera into a shared around-view canvas: '
                         'front=0, right=90, rear=180, left=-90 (or 270).')
    ap.add_argument('--crop-black', action='store_true',
                    help='Crop the surrounding black border after rotation.')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='Scale factor applied after --crop-black (e.g. 0.5 = half size, '
                         '2.0 = double size). Applied only when > 0.')
    ap.add_argument('--grid', action='store_true',
                    help='Overlay a metric grid on the BEV image')
    ap.add_argument('--grid-interval', type=float, default=5.0,
                    help='Grid line interval in metres')

    args = ap.parse_args()

    # ── Load image ────────────────────────────────────────────
    img = cv2.imread(args.input)
    if img is None:
        print(f'ERROR: cannot read {args.input}')
        sys.exit(1)

    h_img, w_img = img.shape[:2]
    if (w_img, h_img) != (args.img_w, args.img_h):
        print(f'WARNING: image is {w_img}×{h_img} but '
              f'--img-w/--img-h is {args.img_w}×{args.img_h}; '
              f'updating to match.')
        args.img_w, args.img_h = w_img, h_img

    # ── Build intrinsics ──────────────────────────────────────
    K = build_K(args.fov, args.img_w, args.img_h)
    fx = K[0, 0]
    print(f'Input  : {args.input}  ({w_img}×{h_img})')
    print(f'K      : f={fx:.1f} px   cx={K[0,2]:.1f}  cy={K[1,2]:.1f}')
    print(f'Extrinsics : height={args.cam_height} m  '
          f'pitch={args.cam_pitch}°  roll={args.cam_roll}°  yaw={args.cam_yaw}°')
    print(f'BEV range  : X={args.x_range} m  Y={args.y_range} m  '
          f'res={args.res} m/px')

    # ── IPM ───────────────────────────────────────────────────
    bev = ipm(img, K,
              cam_height=args.cam_height,
              yaw_deg=args.cam_yaw,
              pitch_deg=args.cam_pitch,
              roll_deg=args.cam_roll,
              x_range=tuple(args.x_range),
              y_range=tuple(args.y_range),
              res=args.res)

    bev_h, bev_w = bev.shape[:2]
    print(f'BEV size   : {bev_w}×{bev_h} px')

    # ── Trim top ──────────────────────────────────────────────
    if args.trim_top > 0:
        if args.trim_top >= bev_h:
            print(f'ERROR: --trim-top ({args.trim_top}) >= BEV height ({bev_h})')
            sys.exit(1)
        bev = bev[args.trim_top:, :]
        trimmed_m = args.trim_top * args.res
        x_range_trimmed = (args.x_range[0],
                           args.x_range[1] - trimmed_m)
        print(f'Trim top   : {args.trim_top} px  ({trimmed_m:.2f} m)  '
              f'→ BEV {bev.shape[1]}×{bev.shape[0]} px  '
              f'x_range now {x_range_trimmed}')
    else:
        x_range_trimmed = tuple(args.x_range)

    if args.grid:
        bev = draw_grid(bev,
                        x_range=x_range_trimmed,
                        y_range=tuple(args.y_range),
                        res=args.res,
                        interval_m=args.grid_interval)

    # ── Rotate ────────────────────────────────────────────────
    if args.rotate != 0.0:
        # Camera origin in BEV pixel coords (vehicle position)
        origin_col = int(round(args.y_range[1] / args.res))
        origin_row = int(round(x_range_trimmed[1] / args.res))
        bev = rotate_bev(bev, args.rotate, origin_col, origin_row)
        print(f'Rotate     : {args.rotate}°  '
              f'(origin: col={origin_col}, row={origin_row})'
              f'  →  BEV {bev.shape[1]}×{bev.shape[0]} px')

    # ── Crop black border ────────────────────────────────────
    if args.crop_black:
        before = (bev.shape[1], bev.shape[0])
        bev = crop_black(bev)
        print(f'Crop black : {before[0]}×{before[1]}'
              f'  →  {bev.shape[1]}×{bev.shape[0]} px')

    # ── Scale ─────────────────────────────────────────────────
    if args.scale != 1.0 and args.scale > 0:
        new_w = max(1, int(round(bev.shape[1] * args.scale)))
        new_h = max(1, int(round(bev.shape[0] * args.scale)))
        interp = cv2.INTER_AREA if args.scale < 1.0 else cv2.INTER_LINEAR
        bev = cv2.resize(bev, (new_w, new_h), interpolation=interp)
        print(f'Scale      : ×{args.scale}  →  {new_w}×{new_h} px')

    # ── Save ──────────────────────────────────────────────────
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(base_dir, f'{stem}_bev.png')

    cv2.imwrite(args.output, bev)
    print(f'Saved  → {args.output}')


if __name__ == '__main__':
    main()
