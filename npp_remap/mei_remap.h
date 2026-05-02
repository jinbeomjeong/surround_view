#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MeiRemapCtx MeiRemapCtx;

/**
 * Allocate GPU resources and upload remap maps (called once per camera).
 *
 * map_x, map_y  : float32 arrays of shape [dst_h * dst_w],
 *                 precomputed by build_undistort_maps() in Python.
 * Returns NULL on failure.
 */
MeiRemapCtx* mei_remap_create(
    int src_w, int src_h,
    int dst_w, int dst_h,
    const float* map_x,
    const float* map_y
);

/**
 * Remap one RGBA frame (CPU → GPU → nppiRemap → CPU).
 *
 * src : src_w * src_h * 4  bytes (RGBA, row-major, contiguous)
 * dst : dst_w * dst_h * 4  bytes (RGBA, row-major, pre-allocated)
 * Returns 0 on success, -1 on error.
 */
int mei_remap_apply(MeiRemapCtx* ctx,
                    const uint8_t* src,
                    uint8_t*       dst);

/** Free all GPU resources. */
void mei_remap_destroy(MeiRemapCtx* ctx);

#ifdef __cplusplus
}
#endif
