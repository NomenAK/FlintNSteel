// Standalone EXL3 API: QTIP weights and CUDA-graph-capturable launches, no torch dependency.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "vendor/quant/exl3_moe_coop.cuh"
namespace fns::exl3 {
struct Linear {
    const uint16_t *trellis = nullptr;
    const half *suh = nullptr;
    const half *svh = nullptr;
    int k = 0, n = 0, bits = 4;
};
// Sets function attributes and occupancy before any CUDA graph capture.
void initialize();
void gemv(const Linear &w, const half *x, int m, half *xh, void *y, bool y_f32, int *locks, cudaStream_t stream);
void moe(MoeCoopParams p, cudaStream_t stream);
struct Router {
    const half *weight = nullptr;
    signed char *hi = nullptr;
    float *scales = nullptr;
};
struct RouterScratch {
    signed char *hi = nullptr, *lo = nullptr;
    float *scales = nullptr, *partial = nullptr;
    half *scores = nullptr;
};
void prepare_router(const Router &w, cudaStream_t stream);
void route(const Router &w, const half *x, int rows, const RouterScratch &s, int64_t *sel, half *rw,
           cudaStream_t stream);
} // namespace fns::exl3
