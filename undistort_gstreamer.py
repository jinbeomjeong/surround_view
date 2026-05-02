#!/usr/bin/env python3
"""
undistort_gstreamer.py  –  MEI fisheye undistortion in a GStreamer pipeline.

Architecture (appsink / appsrc bridge):

  Source:
    multifilesrc → pngdec → videorate → videoconvert → appsink
    (test mode: filesrc → pngdec → imagefreeze → videoconvert → appsink)

  Python bridge (GLib callback, per camera):
    pull sample → np.frombuffer → remap (NPP or cv2) → Gst.Buffer → appsrc

  Sink:
    appsrc → videoconvert → compositor → autovideosink

Remap backends (auto-selected, or forced with --backend):
  npp : nppiRemap_8u_C4R  (GPU, CUDA 13.1 + NPP)   ~0.9 ms  H2D+remap+D2H
  cpu : cv2.remap          (CPU, OpenCV)             ~0.3 ms

Usage:
  source venv/bin/activate
  python undistort_gstreamer.py --test               # quick test, auto backend
  python undistort_gstreamer.py --test --backend npp # force NPP
  python undistort_gstreamer.py --test --backend cpu # force CPU
  python undistort_gstreamer.py --fov 120 --out-size 900
"""

import sys
import os
import time
import argparse

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from undistort_fisheye import load_mei_params, build_undistort_maps


def _try_load_npp():
    """Return NppRemap class if libmei_remap.so is available, else None."""
    try:
        from npp_remap.mei_remap import NppRemap
        return NppRemap
    except Exception as e:
        print(f'  NPP not available ({e}), falling back to CPU')
        return None


CAMERAS   = ('02', '03')
IN_W = IN_H = 1400      # KITTI-360 fisheye resolution
BRIDGE_FMT  = 'RGBA'    # 4-byte aligned; accepted by videoconvert


# ──────────────────────────────────────────────────────────────
# FPS counter
# ──────────────────────────────────────────────────────────────

class FpsCounter:
    def __init__(self, cam_id: str, report_every: int = 30):
        self.cam_id       = cam_id
        self.report_every = report_every
        self._ts: list    = []
        self._n           = 0

    def tick(self) -> None:
        self._ts.append(time.monotonic())
        self._n += 1
        if len(self._ts) > self.report_every:
            self._ts.pop(0)
        if self._n % self.report_every == 0 and len(self._ts) >= 2:
            fps = (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])
            print(f'  cam{self.cam_id}: {fps:.1f} fps  '
                  f'(frame {self._n})')


# ──────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────

class MEIUndistortPipeline:
    """
    GStreamer pipeline that intercepts fisheye frames at appsink,
    applies MEI undistortion with cv2.remap, and re-injects the
    corrected frames via appsrc.

    The remap maps are computed once from the calibration YAML files
    (calibration/image_02.yaml, calibration/image_03.yaml).
    """

    def __init__(
        self,
        paths:     dict,          # {'02': path_pattern, '03': path_pattern}
        calib_dir: str,
        test_mode: bool  = False,
        fps:       int   = 30,
        out_w:     int   = 700,
        out_h:     int   = 700,
        fov_deg:   float = 90.0,
        backend:   str   = 'auto',   # 'auto' | 'npp' | 'cpu'
    ):
        self.paths     = paths
        self.test_mode = test_mode
        self.fps       = fps
        self.out_w     = out_w
        self.out_h     = out_h
        self._appsrcs: dict = {}
        self._fps:     dict = {c: FpsCounter(c) for c in CAMERAS}

        # ── Precompute remap maps (once) ───────────────────────
        # Stored as (map_x, map_y) tuples for cpu path;
        # NppRemap objects hold their own GPU copy.
        self._remappers: dict = {}
        NppRemap = _try_load_npp() if backend != 'cpu' else None
        use_npp  = NppRemap is not None and backend != 'cpu'

        for cam_id in CAMERAS:
            params = load_mei_params(
                os.path.join(calib_dir, f'image_{cam_id}.yaml'))
            map_x, map_y = build_undistort_maps(params, out_w, out_h, fov_deg)

            if use_npp:
                self._remappers[cam_id] = NppRemap(
                    map_x, map_y,
                    src_w=IN_W, src_h=IN_H,
                    dst_w=out_w, dst_h=out_h,
                )
            else:
                self._remappers[cam_id] = (map_x, map_y)

            print(f'  cam {cam_id}: maps ready '
                  f'(xi={params["xi"]:.3f}  '
                  f'FOV={fov_deg}°  out={out_w}×{out_h}  '
                  f'backend={"npp" if use_npp else "cpu"})')

        Gst.init(None)
        self.pipeline = self._build_pipeline()
        self.loop     = GLib.MainLoop()

        # ── Wire appsink callbacks ─────────────────────────────
        for cam_id in CAMERAS:
            sink = self.pipeline.get_by_name(f'sink{cam_id}')
            src  = self.pipeline.get_by_name(f'src{cam_id}')
            if sink is None or src is None:
                raise RuntimeError(
                    f'Element sink{cam_id} or src{cam_id} not found in pipeline.')
            self._appsrcs[cam_id] = src
            sink.connect('new-sample', self._on_new_sample, cam_id)

    # ── Pipeline construction ──────────────────────────────────

    def _build_pipeline(self):
        fps, out_w, out_h = self.fps, self.out_w, self.out_h

        # Caps that appsrc advertises to the downstream pipeline
        src_caps = (
            f'video/x-raw,format={BRIDGE_FMT},'
            f'width={out_w},height={out_h},'
            f'framerate={fps}/1'
        )

        def source_branch(cam_id: str) -> str:
            path = self.paths[cam_id]
            if self.test_mode:
                # Repeat a single image using imagefreeze
                return (
                    f'filesrc location="{path}" ! pngdec ! '
                    f'imagefreeze ! video/x-raw,framerate={fps}/1 ! '
                    f'videoconvert ! '
                    f'video/x-raw,format={BRIDGE_FMT},width={IN_W},height={IN_H} ! '
                    f'appsink name=sink{cam_id} '
                    f'  emit-signals=true max-buffers=1 drop=true sync=false'
                )
            else:
                return (
                    f'multifilesrc location="{path}" '
                    f'  caps="image/png,framerate={fps}/1" ! '
                    f'pngdec ! videorate ! video/x-raw,framerate={fps}/1 ! '
                    f'videoconvert ! '
                    f'video/x-raw,format={BRIDGE_FMT},width={IN_W},height={IN_H} ! '
                    f'appsink name=sink{cam_id} '
                    f'  emit-signals=true max-buffers=1 drop=true sync=false'
                )

        def appsrc_branch(cam_id: str, downstream: str) -> str:
            # format=time: timestamps in nanoseconds (GST_FORMAT_TIME)
            # do-timestamp=false: we copy PTS from the input buffer manually
            return (
                f'appsrc name=src{cam_id} caps="{src_caps}" '
                f'  format=time do-timestamp=false is-live=true ! '
                f'{downstream}'
            )

        # compositor places cam02 on the left, cam03 on the right
        display = (
            f'compositor name=comp '
            f'  sink_0::xpos=0     sink_0::ypos=0 '
            f'  sink_1::xpos={out_w} sink_1::ypos=0 ! '
            f'video/x-raw,width={out_w * 2},height={out_h} ! '
            f'videoconvert ! autovideosink sync=false'
        )
        branches = [
            source_branch('02'),
            source_branch('03'),
            appsrc_branch('02', 'videoconvert ! comp.sink_0'),
            appsrc_branch('03', 'videoconvert ! comp.sink_1'),
            display,
        ]

        pipeline_str = '  '.join(branches)
        print(f'\nPipeline string:\n{pipeline_str}\n')

        try:
            return Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            print(f'Pipeline parse failed: {e.message}')
            raise

    # ── Per-frame processing ───────────────────────────────────

    def _on_new_sample(self, appsink, cam_id: str) -> int:
        """
        GLib callback: pull one frame, undistort it, push to appsrc.
        Runs in the GLib main-loop thread — keep it fast.
        """
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR

        in_buf = sample.get_buffer()
        caps   = sample.get_caps()
        s      = caps.get_structure(0)
        w, h   = s.get_value('width'), s.get_value('height')

        ok, info = in_buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            frame     = np.frombuffer(info.data, np.uint8).reshape(h, w, 4)
            remapper  = self._remappers[cam_id]
            if isinstance(remapper, tuple):
                # CPU path: cv2.remap
                map_x, map_y = remapper
                undistorted  = cv2.remap(
                    frame, map_x, map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            else:
                # NPP path: GPU bilinear remap
                undistorted = remapper.apply(frame)
        finally:
            in_buf.unmap(info)

        # Wrap result in a GstBuffer, preserving input timestamps for sync
        raw     = undistorted.tobytes()
        out_buf = Gst.Buffer.new_allocate(None, len(raw), None)
        out_buf.fill(0, raw)
        out_buf.pts      = in_buf.pts
        out_buf.dts      = in_buf.dts
        out_buf.duration = in_buf.duration

        self._appsrcs[cam_id].emit('push-buffer', out_buf)
        self._fps[cam_id].tick()

        return Gst.FlowReturn.OK

    # ── Lifecycle ──────────────────────────────────────────────

    def run(self) -> None:
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self._on_bus_message)

        self.pipeline.set_state(Gst.State.PLAYING)
        print('Pipeline running — Ctrl-C to stop\n')
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print('\nStopped by user')
        finally:
            self.pipeline.set_state(Gst.State.NULL)

    def _on_bus_message(self, bus, msg) -> None:
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


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description='MEI fisheye undistortion in a GStreamer pipeline (CPU)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--dataset',
        default='/home/jinbeom/workspace/2013_05_28_drive_0004_sync',
        help='Root of the KITTI-360 drive directory (ignored in --test mode)')
    ap.add_argument('--test', action='store_true',
        help='Test mode: loop local image_02.png / image_03.png via imagefreeze')
    ap.add_argument('--fps',      type=int,   default=30,
        help='Pipeline framerate')
    ap.add_argument('--fov',      type=float, default=90.0,
        help='Output horizontal FOV in degrees')
    ap.add_argument('--out-size', type=int,   default=700,
        help='Output image size (square pixels)')
    ap.add_argument('--backend', choices=['auto', 'npp', 'cpu'], default='auto',
        help='Remap backend: auto=NPP if available else CPU (default: auto)')
    args = ap.parse_args()

    base_dir  = os.path.dirname(os.path.abspath(__file__))
    calib_dir = os.path.join(base_dir, 'calibration')

    if args.test:
        paths = {c: os.path.join(base_dir, f'image_{c}.png') for c in CAMERAS}
        print('Mode    : test (imagefreeze with local images)')
    else:
        paths = {
            c: os.path.join(
                args.dataset, f'image_{c}', 'data_rgb', 'image_%06d.png')
            for c in CAMERAS
        }
        print('Mode    : dataset playback')

    print(f'Sink    : compositor → autovideosink')
    print(f'Output  : {args.out_size}×{args.out_size}  FOV={args.fov}°  FPS={args.fps}')

    pipe = MEIUndistortPipeline(
        paths     = paths,
        calib_dir = calib_dir,
        test_mode = args.test,
        fps       = args.fps,
        out_w     = args.out_size,
        out_h     = args.out_size,
        fov_deg   = args.fov,
        backend   = args.backend,
    )
    pipe.run()


if __name__ == '__main__':
    main()
