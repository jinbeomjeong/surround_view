#!/usr/bin/env python3
"""
bev.py

DeepStream pipeline:
  PNG sequence
    -> pngdec -> videorate -> queue -> nvvideoconvert -> RGBA NVMM 1400x1400
    -> nvstreammux
    -> nvdspreprocess(config_undistort.txt)
    -> nvdspreprocess(config_ipm_bev.txt)
    -> nvstreamdemux
    -> per-camera top crop / rotate
    -> outmux -> nvmultistreamtiler -> nvvideoconvert -> nveglglessink

The two nvdspreprocess elements use the same custom library source code, but
that library stores its active config in a process-global pointer. The IPM
config therefore points to a project-local shared-object copy named
libmei_ipm.so.
"""

import argparse
import os
import shutil
import sys
import time

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNDISTORT_CONFIG = os.path.join(BASE_DIR, 'nvds_undistort', 'config_undistort.txt')
IPM_CONFIG = os.path.join(BASE_DIR, 'nvds_ipm', 'config_ipm_bev.txt')
CUSTOM_LIB = os.path.join(BASE_DIR, 'nvds_undistort', 'libmei_undistort.so')
IPM_CUSTOM_LIB = os.path.join(BASE_DIR, 'nvds_undistort', 'libmei_ipm.so')

CAMERAS = ('02', '03')

SRC_W, SRC_H = 1400, 1400
IPM_W, IPM_H = 1400, 1400

DEFAULT_IPM_CROP_TOP = 1000
DEFAULT_IPM_ZOOM = 1.75
SOURCE_QUEUE_BUFFERS = 15
BRANCH_QUEUE_BUFFERS = 4
MUX_BUFFER_POOL_SIZE = 6
BATCHED_PUSH_TIMEOUT_US = 1000000

FLIP_METHODS = {
    '02': 1,  # counter-clockwise 90 degrees
    '03': 3,  # clockwise 90 degrees
}


def prepare_ipm_custom_lib() -> None:
    if (not os.path.exists(IPM_CUSTOM_LIB) or
            os.path.getmtime(IPM_CUSTOM_LIB) < os.path.getmtime(CUSTOM_LIB)):
        shutil.copy2(CUSTOM_LIB, IPM_CUSTOM_LIB)


def crop_height(ipm_crop_top: int) -> int:
    return IPM_H - ipm_crop_top


def src_crop(ipm_crop_top: int) -> str:
    return f'0:{ipm_crop_top}:{IPM_W}:{crop_height(ipm_crop_top)}'


def zoom_src_crop(ipm_zoom: float) -> str:
    crop_w = max(1, min(IPM_W, int(round(IPM_W / ipm_zoom))))
    crop_h = max(1, min(IPM_H, int(round(IPM_H / ipm_zoom))))
    crop_x = (IPM_W - crop_w) // 2
    crop_y = (IPM_H - crop_h) // 2
    return f'{crop_x}:{crop_y}:{crop_w}:{crop_h}'


def ipm_zoom_stage(ipm_zoom: float) -> str:
    if abs(ipm_zoom - 1.0) < 1e-6:
        return ''
    return (
        f'nvvideoconvert src-crop="{zoom_src_crop(ipm_zoom)}" ! '
        f'video/x-raw(memory:NVMM),format=RGBA,width={IPM_W},height={IPM_H} ! '
    )


def build_source_branch(paths: dict, cam_id: str, mux_pad: str,
                        fps: int) -> str:
    path = paths[cam_id]
    return (
        f'multifilesrc location="{path}" '
        f'  caps="image/png,framerate={fps}/1" ! '
        f'pngdec ! videorate ! video/x-raw,framerate={fps}/1 ! '
        f'queue max-size-buffers={SOURCE_QUEUE_BUFFERS} '
        f'max-size-bytes=0 max-size-time=0 ! '
        f'nvvideoconvert ! '
        f'video/x-raw(memory:NVMM),format=RGBA,width={SRC_W},height={SRC_H} ! '
        f'{mux_pad} '
    )


def build_pipeline(paths: dict, fps: int, ipm_config: str,
                   ipm_crop_top: int, ipm_zoom: float, sync: bool) -> str:
    def source_branch(cam_id: str, mux_pad: str) -> str:
        return build_source_branch(paths, cam_id, mux_pad, fps)

    mux = (
        f'nvstreammux name=mux '
        f'  batch-size=2 width={SRC_W} height={SRC_H} '
        f'  nvbuf-memory-type=0 '
        f'  buffer-pool-size={MUX_BUFFER_POOL_SIZE} '
        f'  sync-inputs=true align-inputs=true sort-batch=true '
        f'  batched-push-timeout={BATCHED_PUSH_TIMEOUT_US} ! '
    )

    undistort = (
        f'nvdspreprocess config-file="{UNDISTORT_CONFIG}" ! '
    )

    ipm = (
        f'nvdspreprocess config-file="{ipm_config}" ! '
        f'nvstreamdemux name=demux '
    )

    def post_branch(cam_id: str, demux_pad: str, mux_pad: str) -> str:
        branch_w = IPM_W
        branch_h = crop_height(ipm_crop_top)

        return (
            f'{demux_pad} ! queue max-size-buffers={BRANCH_QUEUE_BUFFERS} '
            f'max-size-bytes=0 max-size-time=0 ! '
            f'{ipm_zoom_stage(ipm_zoom)}'
            f'nvvideoconvert src-crop="{src_crop(ipm_crop_top)}" ! '
            f'video/x-raw(memory:NVMM),format=RGBA,'
            f'width={branch_w},height={branch_h} ! '
            f'nvvideoconvert flip-method={FLIP_METHODS[cam_id]} ! '
            f'video/x-raw(memory:NVMM),format=RGBA,'
            f'width={branch_h},height={branch_w} ! '
            f'{mux_pad} '
        )

    branch_w = IPM_W
    branch_h = crop_height(ipm_crop_top)
    out_w = branch_h
    out_h = branch_w
    tiler_w = out_w * 2
    tiler_h = out_h

    out_mux = (
        f'nvstreammux name=outmux '
        f'  batch-size=2 width={out_w} height={out_h} '
        f'  nvbuf-memory-type=0 '
        f'  buffer-pool-size={MUX_BUFFER_POOL_SIZE} '
        f'  sync-inputs=true align-inputs=true sort-batch=true '
        f'  batched-push-timeout={BATCHED_PUSH_TIMEOUT_US} ! '
    )

    display = (
        f'nvmultistreamtiler rows=1 columns=2 '
        f'  width={tiler_w} height={tiler_h} ! '
        f'video/x-raw(memory:NVMM),format=RGBA,width={tiler_w},height={tiler_h} ! '
        f'nvvideoconvert ! '
        f'nveglglessink sync={str(sync)} max-lateness=-1 qos=false'
    )

    return (
        source_branch('02', 'mux.sink_0') +
        source_branch('03', 'mux.sink_1') +
        mux + undistort + ipm +
        post_branch('02', 'demux.src_0', 'outmux.sink_0') +
        post_branch('03', 'demux.src_1', 'outmux.sink_1') +
        out_mux + display
    )


class FpsMeter:
    def __init__(self, report_interval: float = 2.0):
        self._count = 0
        self._t0 = time.monotonic()
        self._interval = report_interval

    def probe_cb(self, pad, info) -> int:
        self._count += 1
        now = time.monotonic()
        dt = now - self._t0
        if dt >= self._interval:
            fps = self._count / dt
            print(f'  actual FPS: {fps:.1f}  ({self._count} frames / {dt:.1f} s)')
            self._count = 0
            self._t0 = now
        return Gst.PadProbeReturn.OK


class NvdsUndistortIpmCropPipeline:
    def __init__(self, paths: dict, fps: int, ipm_crop_top: int,
                 ipm_zoom: float, sync: bool, print_pipeline: bool):
        Gst.init(None)
        prepare_ipm_custom_lib()
        pipeline_str = build_pipeline(paths, fps, IPM_CONFIG,
                                      ipm_crop_top, ipm_zoom, sync)

        if print_pipeline:
            print(pipeline_str)

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            print(f'Pipeline parse failed: {e.message}')
            raise

        tiler = self.pipeline.get_by_name('nvmultistreamtiler0')
        if tiler:
            src_pad = tiler.get_static_pad('src')
            self._fps_meter = FpsMeter(report_interval=2.0)
            src_pad.add_probe(Gst.PadProbeType.BUFFER, self._fps_meter.probe_cb)

        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self._on_message)

    def run(self) -> None:
        self.pipeline.set_state(Gst.State.PLAYING)
        print('Undistort + IPM crop pipeline running — Ctrl-C to stop\n')
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
            w, dbg = msg.parse_warning()
            print(f'GStreamer warning: {w.message}')
            if dbg:
                print(f'  debug: {dbg}')


def main() -> None:
    ap = argparse.ArgumentParser(
        description='DeepStream chained undistort + IPM BEV crop pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        '--dataset',
        default='/home/jinbeom/workspace/2013_05_28_drive_0004_sync',
        help='Root of KITTI-360 drive',
    )
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument(
        '--ipm-crop-top',
        type=int,
        default=DEFAULT_IPM_CROP_TOP,
        help='Top coordinate of the crop rectangle in the 1400x1400 IPM image '
             'before per-camera rotation',
    )
    ap.add_argument('--no-sync', action='store_true',
                    help='Disable synchronized display on nveglglessink')
    ap.add_argument('--print-pipeline', action='store_true',
                    help='Print the generated GStreamer pipeline string')
    args = ap.parse_args()

    for path in (UNDISTORT_CONFIG, IPM_CONFIG, CUSTOM_LIB):
        if not os.path.exists(path):
            print(f'Required file not found: {path}')
            sys.exit(1)

    ipm_crop_top = args.ipm_crop_top
    if ipm_crop_top < 0 or ipm_crop_top >= IPM_H:
        print(f'Invalid --ipm-crop-top: {ipm_crop_top} '
              f'(expected 0..{IPM_H - 1})')
        sys.exit(1)
    if DEFAULT_IPM_ZOOM < 1.0:
        print(f'Invalid DEFAULT_IPM_ZOOM: {DEFAULT_IPM_ZOOM} (expected >= 1.0)')
        sys.exit(1)

    paths = {
        c: os.path.join(
            args.dataset, f'image_{c}', 'data_rgb', 'image_%06d.png')
        for c in CAMERAS
    }

    print('Mode   : dataset playback')
    print(f'FPS    : {args.fps}')
    print(f'Input  : {SRC_W}x{SRC_H} PNG sequences')
    print(f'Stages : undistort {SRC_W}x{SRC_H} -> IPM {IPM_W}x{IPM_H}')
    print(f'Zoom   : {DEFAULT_IPM_ZOOM:.3g}x center crop "{zoom_src_crop(DEFAULT_IPM_ZOOM)}"'
          f' -> {IPM_W}x{IPM_H}')
    branch_w = IPM_W
    branch_h = crop_height(ipm_crop_top)
    print(f'Crop   : top-only src-crop "{src_crop(ipm_crop_top)}"'
          f' -> {branch_w}x{branch_h}')
    print(f'Rotate : flip-methods {FLIP_METHODS}')
    out_w = branch_h
    out_h = branch_w
    print(f'Output : {out_w * 2}x{out_h} px  ({out_w}x{out_h} per camera)\n')

    NvdsUndistortIpmCropPipeline(
        paths, args.fps, ipm_crop_top, DEFAULT_IPM_ZOOM,
        not args.no_sync, args.print_pipeline).run()


if __name__ == '__main__':
    main()
