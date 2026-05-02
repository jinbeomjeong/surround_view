#include "mei_remap.h"

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_geometry_transforms.h>

#include <cstdio>
#include <new>

// ── helpers ───────────────────────────────────────────────────

#define CHECK_CUDA(call, msg) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "[mei_remap] CUDA error (%s): %s\n", \
                    (msg), cudaGetErrorString(_e)); \
            goto fail; \
        } \
    } while (0)

#define CHECK_NPP(call) \
    do { \
        NppStatus _s = (call); \
        if (_s != NPP_SUCCESS) { \
            fprintf(stderr, "[mei_remap] NPP error: %d\n", (int)_s); \
            return -1; \
        } \
    } while (0)

// ── context struct ────────────────────────────────────────────

struct MeiRemapCtx {
    int src_w, src_h;
    int dst_w, dst_h;

    cudaStream_t    stream;
    NppStreamContext npp_ctx;

    uint8_t* d_src;     // device: source frame  (src_w * src_h * 4 bytes)
    uint8_t* d_dst;     // device: output frame  (dst_w * dst_h * 4 bytes)
    float*   d_map_x;   // device: remap map X   (dst_w * dst_h * sizeof(float))
    float*   d_map_y;   // device: remap map Y   (dst_w * dst_h * sizeof(float))
};

// ── public API ────────────────────────────────────────────────

MeiRemapCtx* mei_remap_create(
    int src_w, int src_h,
    int dst_w, int dst_h,
    const float* map_x,
    const float* map_y)
{
    MeiRemapCtx* ctx = new (std::nothrow) MeiRemapCtx{};
    if (!ctx) return nullptr;

    ctx->src_w = src_w;  ctx->src_h = src_h;
    ctx->dst_w = dst_w;  ctx->dst_h = dst_h;
    ctx->d_src = ctx->d_dst = nullptr;
    ctx->d_map_x = ctx->d_map_y = nullptr;

    // ── CUDA stream ──────────────────────────────────────────
    CHECK_CUDA(cudaStreamCreate(&ctx->stream), "streamCreate");

    // ── NppStreamContext from current device + stream ────────
    {
        int dev;
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDevice(&dev), "getDevice");
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev), "getDeviceProps");

        ctx->npp_ctx = {};
        ctx->npp_ctx.hStream                        = ctx->stream;
        ctx->npp_ctx.nCudaDeviceId                  = dev;
        ctx->npp_ctx.nMultiProcessorCount           = prop.multiProcessorCount;
        ctx->npp_ctx.nMaxThreadsPerMultiProcessor   = prop.maxThreadsPerMultiProcessor;
        ctx->npp_ctx.nMaxThreadsPerBlock            = prop.maxThreadsPerBlock;
        ctx->npp_ctx.nSharedMemPerBlock             = prop.sharedMemPerBlock;
        ctx->npp_ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
        ctx->npp_ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    }

    // ── device allocations ───────────────────────────────────
    CHECK_CUDA(cudaMalloc(&ctx->d_src,   (size_t)src_w * src_h * 4),          "malloc d_src");
    CHECK_CUDA(cudaMalloc(&ctx->d_dst,   (size_t)dst_w * dst_h * 4),          "malloc d_dst");
    CHECK_CUDA(cudaMalloc(&ctx->d_map_x, (size_t)dst_w * dst_h * sizeof(float)), "malloc d_map_x");
    CHECK_CUDA(cudaMalloc(&ctx->d_map_y, (size_t)dst_w * dst_h * sizeof(float)), "malloc d_map_y");

    // ── upload remap maps once ───────────────────────────────
    CHECK_CUDA(cudaMemcpy(ctx->d_map_x, map_x,
                          (size_t)dst_w * dst_h * sizeof(float),
                          cudaMemcpyHostToDevice), "upload map_x");
    CHECK_CUDA(cudaMemcpy(ctx->d_map_y, map_y,
                          (size_t)dst_w * dst_h * sizeof(float),
                          cudaMemcpyHostToDevice), "upload map_y");

    fprintf(stderr, "[mei_remap] created  src=%dx%d  dst=%dx%d  dev=%d (sm_%d%d)\n",
            src_w, src_h, dst_w, dst_h,
            ctx->npp_ctx.nCudaDeviceId,
            ctx->npp_ctx.nCudaDevAttrComputeCapabilityMajor,
            ctx->npp_ctx.nCudaDevAttrComputeCapabilityMinor);
    return ctx;

fail:
    mei_remap_destroy(ctx);
    return nullptr;
}


int mei_remap_apply(MeiRemapCtx* ctx,
                    const uint8_t* src,
                    uint8_t*       dst)
{
    if (!ctx) return -1;

    const size_t src_bytes = (size_t)ctx->src_w * ctx->src_h * 4;
    const size_t dst_bytes = (size_t)ctx->dst_w * ctx->dst_h * 4;

    // H2D: upload source frame
    if (cudaMemcpyAsync(ctx->d_src, src, src_bytes,
                        cudaMemcpyHostToDevice, ctx->stream) != cudaSuccess)
        return -1;

    // GPU: nppiRemap RGBA 8-bit 4-channel with bilinear interpolation
    NppiSize src_size = { ctx->src_w, ctx->src_h };
    NppiRect src_roi  = { 0, 0, ctx->src_w, ctx->src_h };
    NppiSize dst_size = { ctx->dst_w, ctx->dst_h };

    CHECK_NPP(nppiRemap_8u_C4R_Ctx(
        ctx->d_src,               src_size,
        ctx->src_w * 4,           src_roi,
        ctx->d_map_x,             ctx->dst_w * (int)sizeof(float),
        ctx->d_map_y,             ctx->dst_w * (int)sizeof(float),
        ctx->d_dst,               ctx->dst_w * 4,
        dst_size,
        NPPI_INTER_LINEAR,
        ctx->npp_ctx
    ));

    // D2H: download result
    if (cudaMemcpyAsync(dst, ctx->d_dst, dst_bytes,
                        cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess)
        return -1;

    // Wait for stream to finish
    if (cudaStreamSynchronize(ctx->stream) != cudaSuccess)
        return -1;

    return 0;
}


void mei_remap_destroy(MeiRemapCtx* ctx)
{
    if (!ctx) return;
    if (ctx->d_src)   cudaFree(ctx->d_src);
    if (ctx->d_dst)   cudaFree(ctx->d_dst);
    if (ctx->d_map_x) cudaFree(ctx->d_map_x);
    if (ctx->d_map_y) cudaFree(ctx->d_map_y);
    if (ctx->stream)  cudaStreamDestroy(ctx->stream);
    delete ctx;
}
