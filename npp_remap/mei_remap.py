"""
mei_remap.py  –  Python ctypes wrapper for libmei_remap.so

Replaces cv2.remap with nppiRemap (NPP, GPU bilinear interpolation).
The remap maps are uploaded to GPU once at construction time.

Usage:
    from npp_remap.mei_remap import NppRemap

    remap = NppRemap(map_x, map_y, src_w=1400, src_h=1400,
                                   dst_w=700,  dst_h=700)
    undistorted = remap.apply(frame_rgba)   # numpy uint8 RGBA → numpy uint8 RGBA
    remap.close()                            # or: use as context manager
"""

import ctypes
import os
import numpy as np


_LIB_PATH = os.path.join(os.path.dirname(__file__), 'libmei_remap.so')


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(_LIB_PATH)

    # void* mei_remap_create(int,int,int,int, float*, float*)
    lib.mei_remap_create.restype  = ctypes.c_void_p
    lib.mei_remap_create.argtypes = [
        ctypes.c_int, ctypes.c_int,   # src_w, src_h
        ctypes.c_int, ctypes.c_int,   # dst_w, dst_h
        ctypes.POINTER(ctypes.c_float),  # map_x
        ctypes.POINTER(ctypes.c_float),  # map_y
    ]

    # int mei_remap_apply(void*, uint8*, uint8*)
    lib.mei_remap_apply.restype  = ctypes.c_int
    lib.mei_remap_apply.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),  # src
        ctypes.POINTER(ctypes.c_uint8),  # dst
    ]

    # void mei_remap_destroy(void*)
    lib.mei_remap_destroy.restype  = None
    lib.mei_remap_destroy.argtypes = [ctypes.c_void_p]

    return lib


_lib: ctypes.CDLL | None = None

def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_lib()
    return _lib


class NppRemap:
    """
    GPU-accelerated fisheye undistortion using NPP nppiRemap_8u_C4R.

    map_x, map_y : float32 numpy arrays of shape (dst_h, dst_w),
                   precomputed by build_undistort_maps().
    """

    def __init__(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        src_w: int, src_h: int,
        dst_w: int, dst_h: int,
    ):
        lib = _get_lib()

        # Ensure C-contiguous float32
        mx = np.ascontiguousarray(map_x, dtype=np.float32)
        my = np.ascontiguousarray(map_y, dtype=np.float32)

        self._ctx = lib.mei_remap_create(
            src_w, src_h, dst_w, dst_h,
            mx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            my.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if not self._ctx:
            raise RuntimeError('mei_remap_create failed — check CUDA device availability')

        self._dst_w = dst_w
        self._dst_h = dst_h
        self._lib   = lib

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Remap one RGBA frame.

        frame : uint8 numpy array, shape (src_h, src_w, 4), C-contiguous
        Returns uint8 numpy array, shape (dst_h, dst_w, 4)
        """
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        dst   = np.empty((self._dst_h, self._dst_w, 4), dtype=np.uint8)

        ret = self._lib.mei_remap_apply(
            self._ctx,
            frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if ret != 0:
            raise RuntimeError('mei_remap_apply failed')
        return dst

    def close(self) -> None:
        if self._ctx:
            self._lib.mei_remap_destroy(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()
