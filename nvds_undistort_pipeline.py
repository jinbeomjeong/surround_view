#!/usr/bin/env python3
"""
nvds_undistort_pipeline.py  –  DeepStream MEI fisheye undistortion pipeline.

Pipeline:
  multifilesrc  ─► pngdec ─► videorate ─► nvvideoconvert(→RGBA NVMM) ─► nvstreammux
  multifilesrc  ─► pngdec ─► videorate ─► nvvideoconvert(→RGBA NVMM) ─┘
        ↓
  nvdspreprocess  config-file=config_undistort.txt
    └─ libmei_undistort.so → CustomTransformation():
         d_tmp  ← in_surf[i].dataPtr           (save original, GPU→GPU)
         in_surf[i].dataPtr ← nppiRemap(d_tmp) (undistort for display)
         out_surf[i].dataPtr ← in_surf[i]      (tensor copy)
        ↓
  nvmultistreamtiler  (side-by-side)
        ↓
  nvvideoconvert ─► nveglglessink

Usage:
  source venv/bin/activate
  python nvds_undistort_pipeline.py
  python nvds_undistort_pipeline.py --dataset /path/to/kitti360/drive
  python nvds_undistort_pipeline.py --test          # local images via imagefreeze
  python nvds_undistort_pipeline.py --fps 15
"""

import sys
import os
import time
import argparse

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CONFIG     = os.path.join(BASE_DIR, 'nvds_undistort', 'config_undistort.txt')
CAMERAS    = ('02', '03')


# ──────────────────────────────────────────────────────────────
# Pipeline builder
# ──────────────────────────────────────────────────────────────

def build_pipeline(paths: dict, fps: int, test_mode: bool,
                   src_w: int = 1400, src_h: int = 1400) -> str:
    """
    Return the GStreamer pipeline description string.

    Source branches convert PNG frames to RGBA NVMM so that
    nvstreammux batches RGBA surfaces that nvdspreprocess can
    access directly as CUDA device pointers.
    """

    def source_branch(cam_id: str, mux_pad: str) -> str:
        path = paths[cam_id]
        if test_mode:
            # Single image repeated via imagefreeze
            return (
                f'filesrc location="{path}" ! pngdec ! '
                f'imagefreeze ! video/x-raw,framerate={fps}/1 ! '
                f'nvvideoconvert ! '
                f'video/x-raw(memory:NVMM),format=RGBA,'
                f'width={src_w},height={src_h} ! '
                f'{mux_pad} '
            )
        else:
            return (
                f'multifilesrc location="{path}" '
                f'  caps="image/png,framerate={fps}/1" ! '
                f'pngdec ! videorate ! video/x-raw,framerate={fps}/1 ! '
                f'nvvideoconvert ! '
                f'video/x-raw(memory:NVMM),format=RGBA,'
                f'width={src_w},height={src_h} ! '
                f'{mux_pad} '
            )

    mux = (
        f'nvstreammux name=mux '
        f'  batch-size=2 width={src_w} height={src_h} '
        f'  nvbuf-memory-type=0 ! '          # 0 = NVBUF_MEM_CUDA_DEVICE
    )

    preprocess = (
        f'nvdspreprocess config-file="{CONFIG}" ! '
    )

    tiler_w = src_w * 2
    tiler_h = src_h
    display = (
        f'nvmultistreamtiler rows=1 columns=2 '
        f'  width={tiler_w} height={tiler_h} ! '
        f'nvvideoconvert ! '
        f'nveglglessink sync=True'
    )

    return (
        source_branch('02', 'mux.sink_0') +
        source_branch('03', 'mux.sink_1') +
        mux + preprocess + display
    )


# ──────────────────────────────────────────────────────────────
# FPS counter (pad probe on tiler src pad)
# ──────────────────────────────────────────────────────────────

class FpsMeter:
    """Counts buffers passing through a GStreamer pad and reports FPS."""

    def __init__(self, report_interval: float = 2.0):
        self._count   = 0
        self._t0      = time.monotonic()
        self._interval = report_interval

    def probe_cb(self, pad, info) -> int:
        self._count += 1
        now = time.monotonic()
        dt  = now - self._t0
        if dt >= self._interval:
            fps = self._count / dt
            print(f'  actual FPS: {fps:.1f}  ({self._count} frames / {dt:.1f} s)')
            self._count = 0
            self._t0    = now
        return Gst.PadProbeReturn.OK


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

class NvdsUndistortPipeline:
    def __init__(self, paths: dict, fps: int, test_mode: bool):
        Gst.init(None)
        pipeline_str = build_pipeline(paths, fps, test_mode)

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            print(f'Pipeline parse failed: {e.message}')
            raise

        # Attach FPS probe to nvmultistreamtiler src pad
        tiler = self.pipeline.get_by_name('nvmultistreamtiler0')
        if tiler:
            src_pad = tiler.get_static_pad('src')
            self._fps_meter = FpsMeter(report_interval=2.0)
            src_pad.add_probe(Gst.PadProbeType.BUFFER,
                              self._fps_meter.probe_cb)

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self._on_message)

    def run(self) -> None:
        self.pipeline.set_state(Gst.State.PLAYING)
        print('Pipeline running — Ctrl-C to stop\n')
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print('\nStopped by user')
        finally:
            self.pipeline.set_state(Gst.State.NULL)

    def _on_message(self, bus, msg) -> None:
        t = msg.type
        if t == Gst.MessageType.EOS:
            print('End of stream')
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f'GStreamer error: {err.message}')
            if dbg:
                print(f'  debug: {dbg}')
            self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            w, _ = msg.parse_warning()
            print(f'GStreamer warning: {w.message}')


def main() -> None:
    ap = argparse.ArgumentParser(
        description='DeepStream MEI fisheye undistortion pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--dataset',
        default='/home/jinbeom/workspace/2013_05_28_drive_0004_sync',
        help='Root of KITTI-360 drive (ignored with --test)')
    ap.add_argument('--test', action='store_true',
        help='Test mode: use local image_02.png / image_03.png via imagefreeze')
    ap.add_argument('--fps', type=int, default=10)
    args = ap.parse_args()

    if not os.path.exists(CONFIG):
        print(f'Config not found: {CONFIG}')
        sys.exit(1)

    if args.test:
        paths = {c: os.path.join(BASE_DIR, f'image_{c}.png') for c in CAMERAS}
        print('Mode : test (imagefreeze with local images)')
    else:
        paths = {
            c: os.path.join(
                args.dataset, f'image_{c}', 'data_rgb', 'image_%06d.png')
            for c in CAMERAS
        }
        print('Mode : dataset playback')

    print(f'FPS    : {args.fps}')

    NvdsUndistortPipeline(paths, args.fps, args.test).run()


if __name__ == '__main__':
    main()
