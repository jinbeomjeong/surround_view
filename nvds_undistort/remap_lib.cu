/*
 * remap_lib.cu  –  nvdspreprocess custom remap library
 *
 * MEI fisheye undistortion using nppiRemap directly on NvBufSurface
 * CUDA device pointers (NVMM) — zero CPU↔GPU transfer per frame.
 *
 * Why in-place modification of in_surf:
 *   nvdspreprocess passes the ORIGINAL GstBuffer (in_surf) downstream
 *   to nvmultistreamtiler / nveglglessink.  The out_surf is an internal
 *   tensor buffer consumed by nvinfer, not by the display path.
 *   To see undistorted frames on screen we overwrite in_surf in-place.
 *
 * Per-frame data path:
 *   1. cudaMemcpy  in_surf[i].dataPtr → d_tmp[cam]   (save original)
 *   2. nppiRemap   d_tmp[cam]         → in_surf[i].dataPtr  (display)
 *
 * Both per-frame steps are GPU-only (NVMM pointers throughout).
 *
 * Exported symbols required by nvdspreprocess:
 *   initLib / deInitLib / CustomTransformation /
 *   CustomAsyncTransformation / CustomTensorPreparation
 *
 * user-config keys (config_undistort.txt):
 *   num-cameras        number of camera streams        (default: 2)
 *   src-w, src-h       fisheye source resolution       (default: 1400)
 *   dst-w, dst-h       undistorted output resolution   (default: 1400)
 *   cam-N-map-x        binary float32 remap map_x file
 *   cam-N-map-y        binary float32 remap map_y file
 *
 * Pipeline requirement:
 *   nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA
 *   before nvstreammux so in_surf carries 4-channel RGBA data.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <new>

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_geometry_transforms.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdspreprocess_interface.h"
#include "nvdspreprocess_lib.h"


// ── logging ───────────────────────────────────────────────────

#define LOG(fmt, ...) \
    fprintf(stderr, "[mei_undistort] " fmt "\n", ##__VA_ARGS__)


// ── helpers ───────────────────────────────────────────────────

static std::string cfg_get(const CustomInitParams& p,
                             const std::string& key,
                             const std::string& def = "")
{
    auto it = p.user_configs.find(key);
    return it != p.user_configs.end() ? it->second : def;
}

static bool upload_map(const std::string& path,
                        float** d_out, int w, int h)
{
    const size_t n = (size_t)w * h;
    std::vector<float> buf(n);

    std::ifstream f(path, std::ios::binary);
    if (!f) { LOG("cannot open: %s", path.c_str()); return false; }
    f.read(reinterpret_cast<char*>(buf.data()), n * sizeof(float));
    if ((size_t)f.gcount() != n * sizeof(float)) {
        LOG("short read: %s", path.c_str()); return false;
    }
    if (cudaMalloc(d_out, n * sizeof(float)) != cudaSuccess) {
        LOG("cudaMalloc failed"); return false;
    }
    cudaMemcpy(*d_out, buf.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    LOG("  map uploaded  %s  (%dx%d)", path.c_str(), w, h);
    return true;
}


// ── context ───────────────────────────────────────────────────

struct CamMaps {
    int    src_w, src_h;
    int    dst_w, dst_h;
    float* d_map_x = nullptr;
    float* d_map_y = nullptr;
    uint8_t* d_tmp = nullptr;   // temp buffer: copy of original src frame
};

struct CustomCtx {
    CustomInitParams     initParams;
    std::vector<CamMaps> cam_maps;
    NppStreamContext     npp_ctx;
    cudaStream_t         stream = nullptr;
};

// nvdspreprocess does not pass CustomCtx to CustomTransformation,
// so we keep a module-level pointer set during initLib.
static CustomCtx* g_ctx = nullptr;


// ── exported symbols ──────────────────────────────────────────

extern "C"
CustomCtx* initLib(CustomInitParams initparams)
{
    auto* ctx = new (std::nothrow) CustomCtx();
    if (!ctx) return nullptr;
    ctx->initParams = initparams;

    const int num_cams = std::stoi(cfg_get(initparams, "num-cameras", "2"));
    const int src_w    = std::stoi(cfg_get(initparams, "src-w", "1400"));
    const int src_h    = std::stoi(cfg_get(initparams, "src-h", "1400"));
    const int dst_w    = std::stoi(cfg_get(initparams, "dst-w", "1400"));
    const int dst_h    = std::stoi(cfg_get(initparams, "dst-h", "1400"));

    LOG("init  num_cams=%d  src=%dx%d  dst=%dx%d",
        num_cams, src_w, src_h, dst_w, dst_h);

    // ── CUDA stream ──────────────────────────────────────────
    if (cudaStreamCreate(&ctx->stream) != cudaSuccess) {
        LOG("cudaStreamCreate failed"); goto fail;
    }

    // ── NppStreamContext ─────────────────────────────────────
    {
        int dev; cudaDeviceProp prop;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);

        ctx->npp_ctx = {};
        ctx->npp_ctx.hStream                            = ctx->stream;
        ctx->npp_ctx.nCudaDeviceId                      = dev;
        ctx->npp_ctx.nMultiProcessorCount               = prop.multiProcessorCount;
        ctx->npp_ctx.nMaxThreadsPerMultiProcessor       = prop.maxThreadsPerMultiProcessor;
        ctx->npp_ctx.nMaxThreadsPerBlock                = prop.maxThreadsPerBlock;
        ctx->npp_ctx.nSharedMemPerBlock                 = prop.sharedMemPerBlock;
        ctx->npp_ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
        ctx->npp_ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
        LOG("GPU: %s  sm_%d%d", prop.name, prop.major, prop.minor);
    }

    // ── load remap maps + allocate temp buffers ──────────────
    ctx->cam_maps.resize(num_cams);
    for (int c = 0; c < num_cams; ++c) {
        CamMaps& m  = ctx->cam_maps[c];
        m.src_w = src_w;  m.src_h = src_h;
        m.dst_w = dst_w;  m.dst_h = dst_h;

        const std::string kx = "cam-" + std::to_string(c) + "-map-x";
        const std::string ky = "cam-" + std::to_string(c) + "-map-y";

        if (!upload_map(cfg_get(initparams, kx), &m.d_map_x, dst_w, dst_h) ||
            !upload_map(cfg_get(initparams, ky), &m.d_map_y, dst_w, dst_h))
            goto fail;

        // temp buffer: tight-packed (src_w * 4 stride, no row padding)
        if (cudaMalloc(&m.d_tmp, (size_t)src_w * 4 * src_h) != cudaSuccess) {
            LOG("cudaMalloc d_tmp failed"); goto fail;
        }
    }

    g_ctx = ctx;
    LOG("init complete — %d camera(s) ready", num_cams);
    return ctx;

fail:
    deInitLib(ctx);
    return nullptr;
}


extern "C"
void deInitLib(CustomCtx* ctx)
{
    if (!ctx) return;
    for (auto& m : ctx->cam_maps) {
        if (m.d_map_x) { cudaFree(m.d_map_x); m.d_map_x = nullptr; }
        if (m.d_map_y) { cudaFree(m.d_map_y); m.d_map_y = nullptr; }
        if (m.d_tmp)   { cudaFree(m.d_tmp);   m.d_tmp   = nullptr; }
    }
    if (ctx->stream) { cudaStreamDestroy(ctx->stream); ctx->stream = nullptr; }
    if (g_ctx == ctx) g_ctx = nullptr;
    delete ctx;
}


/*
 * CustomTransformation  —  called once per batch by nvdspreprocess
 *
 * in_surf  : batched RGBA NVMM frames from nvstreammux (src_w × src_h)
 *            → also flows downstream to nvmultistreamtiler / nveglglessink
 * out_surf : pre-allocated NVMM surfaces at dst size (tensor buffer)
 *            → unused in this display-only pipeline
 *
 * We modify in_surf in-place so the DISPLAY path shows undistorted frames.
 *
 * Per-surface steps:
 *   1. d_tmp  ← in_surf[i].dataPtr          (save original, GPU-to-GPU)
 *   2. in_surf[i].dataPtr ← nppiRemap(d_tmp) (undistort for display)
 */
extern "C"
NvDsPreProcessStatus CustomTransformation(NvBufSurface*         in_surf,
                                           NvBufSurface*         /*out_surf*/,
                                           CustomTransformParams& /*params*/)
{
    if (!g_ctx) {
        LOG("context not initialized");
        return NVDSPREPROCESS_CUSTOM_LIB_FAILED;
    }

    const int num_cams = (int)g_ctx->cam_maps.size();

    for (uint32_t i = 0; i < in_surf->numFilled; ++i) {
        NvBufSurfaceParams& inp = in_surf->surfaceList[i];
        CamMaps& maps = g_ctx->cam_maps[i % num_cams];

        const int row_bytes = (int)inp.width * 4;  // tight stride for d_tmp

        // ── Step 1: in_surf → d_tmp  (de-stride to tight packing) ──
        // inp.pitch may include row-padding; d_tmp is allocated tightly.
        cudaMemcpy2DAsync(
            maps.d_tmp,   row_bytes,          // dst (tight)
            inp.dataPtr,  (int)inp.pitch,      // src (NvBufSurface, may be padded)
            row_bytes,    (int)inp.height,     // copy width, rows
            cudaMemcpyDeviceToDevice, g_ctx->stream);

        // ── Step 2: clear destination, then nppiRemap d_tmp → in_surf ──
        // nppiRemap does not guarantee writes for invalid map coordinates.
        // Clear first so out-of-map IPM/undistort regions become black.
        {
            NppiSize src_size = { (int)inp.width,  (int)inp.height };
            NppiRect src_roi  = { 0, 0, (int)inp.width, (int)inp.height };
            NppiSize dst_size = { maps.dst_w, maps.dst_h };

            cudaError_t ce = cudaMemset2DAsync(
                inp.dataPtr, (size_t)inp.pitch,
                0,
                (size_t)maps.dst_w * 4, (size_t)maps.dst_h,
                g_ctx->stream);
            if (ce != cudaSuccess) {
                LOG("cudaMemset2DAsync failed surface %u: %s",
                    i, cudaGetErrorString(ce));
                return NVDSPREPROCESS_CUDA_ERROR;
            }

            NppStatus st = nppiRemap_8u_C4R_Ctx(
                (const Npp8u*)maps.d_tmp, src_size,
                row_bytes,                src_roi,
                maps.d_map_x, maps.dst_w * (int)sizeof(float),
                maps.d_map_y, maps.dst_w * (int)sizeof(float),
                (Npp8u*)inp.dataPtr,      (int)inp.pitch,
                dst_size,
                NPPI_INTER_CUBIC,
                g_ctx->npp_ctx
            );
            if (st != NPP_SUCCESS) {
                LOG("nppiRemap failed surface %u: %d", i, (int)st);
                return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
            }
        }

    }

    if (cudaStreamSynchronize(g_ctx->stream) != cudaSuccess)
        return NVDSPREPROCESS_CUDA_ERROR;

    return NVDSPREPROCESS_SUCCESS;
}


extern "C"
NvDsPreProcessStatus CustomAsyncTransformation(NvBufSurface*         in_surf,
                                                NvBufSurface*         out_surf,
                                                CustomTransformParams& params)
{
    return CustomTransformation(in_surf, out_surf, params);
}


extern "C"
NvDsPreProcessStatus CustomTensorPreparation(CustomCtx*              /*ctx*/,
                                              NvDsPreProcessBatch*    /*batch*/,
                                              NvDsPreProcessCustomBuf*& /*buf*/,
                                              CustomTensorParams&     /*tensorParam*/,
                                              NvDsPreProcessAcquirer* /*acquirer*/)
{
    return NVDSPREPROCESS_TENSOR_NOT_READY;
}
