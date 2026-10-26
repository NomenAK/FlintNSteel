/*
 * Origin: exllamav3 v1.6.2 (4c842731), Flint patch 3b013de9.
 * exllamav3_ext/quant/exl3_{gemv,gemm,moe_coop}.cu: dispatch reduced to Flint / sm_86.
 * Router: exllamav3_ext/routing.cu and routing_gemm.cu, same revision;
 * deterministic projection and top-k without the ATen front end.
 * Copyright (c) 2025 Turboderp
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
// sm_86 dispatch: same kernels and arithmetic as the exllamav3 extension 3b013de9.
#include "exl3_kernels.h"
#include "vendor/util.h"
#include "vendor/util.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "vendor/quant/exl3_gemv_kernel.cuh"
#include "vendor/quant/exl3_gemm_kernel.cuh"
#define EXL3_MOE_COOP_DEFINE_ROT
#include "vendor/quant/exl3_moe_coop_kernel.cuh"
#include "vendor/routing_gemm_kernel.cuh"
#include <algorithm>
#include <map>
namespace fns::exl3 {
#undef MIN
#undef MAX
#undef CEIL_DIVIDE
#undef small
namespace {
int sms = 0;
std::map<void *, int> occupancy;
void *small(bool fp32, int mode, int cfg) {
#define S(F, M, C)                                                                                                     \
    if (fp32 == F && mode == M && cfg == C)                                                                            \
        return (void *)exl3_gemv_kernel<4, F, 2, M, C, false>;
    S(false, 0, 0)
    S(false, 0, 1) S(false, 1, 0) S(false, 1, 1) S(true, 0, 0) S(true, 0, 1) S(true, 1, 0) S(true, 1, 1)
#undef S
        return nullptr;
}
void *large(int bits, bool f) {
#define L(B, F)                                                                                                        \
    if (bits == B && f == F)                                                                                           \
        return (void *)exl3_gemm_kernel<B, false, F, 2, 16, 32, 128, 4, 3>;
    L(4, false)
    L(4, true) L(5, false) L(5, true)
#undef L
        return nullptr;
}
} // namespace
void initialize() {
    if (sms)
        return;
    int device;
    cuda_check(cudaGetDevice(&device));
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, device));
    TORCH_CHECK(prop.major == 8 && prop.minor == 6, "EXL3: sm_86 required");
    sms = prop.multiProcessorCount;
    cuda_check(
        cudaFuncSetAttribute((void *)routing_gemm_i8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, RG_SMEM));
    for (bool f : {false, true})
        for (int m : {0, 1})
            for (int c : {0, 1}) {
                void *p = small(f, m, c);
                int occ;
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p, c ? 256 : 512, 0));
                occupancy[p] = occ;
            }
    for (int b : {4, 5})
        for (bool f : {false, true}) {
            void *p = large(b, f);
            cuda_check(cudaFuncSetAttribute(p, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
        }
    using namespace exl3_moe_coop_ns;
    const int sa = smem_a_bytes<4>(2048), sb = smem_b_bytes<4>();
    if (sa > 48 * 1024)
        cuda_check(cudaFuncSetAttribute((void *)exl3_moe_coop_a_kernel<4, 2, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, sa));
    if (sb > 48 * 1024)
        cuda_check(cudaFuncSetAttribute((void *)exl3_moe_coop_b_kernel<4, 2, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, sb));
}
void gemv(const Linear &w, const half *x, int m, half *xh, void *y, bool f, int *locks, cudaStream_t stream) {
    TORCH_CHECK(sms && m >= 1 && m <= 8 && w.k % 128 == 0 && w.n % 128 == 0, "EXL3 GEMV: unsupported shape");
    const half *su = w.suh, *sv = w.svh;
    const uint16_t *b = w.trellis;
    int k = w.k, n = w.n;
    void *args[] = {&x, &b, &y, &m, &k, &n, &locks, &su, &xh, &sv};
    void *p = nullptr;
    int grid = sms, threads = 512, smem = SMEM_MAX;
    if (w.bits == 4) {
        void *narrow = small(f, m == 1 ? 0 : 1, 0);
        int cfg = -1;
        if (n / 32 <= occupancy.at(narrow) * sms || (k <= 2048 && n <= 8192))
            cfg = 0;
        else if ((n >= 8192 && k <= 4096) || (n >= 8192 && n <= 10240 && k <= 5120))
            cfg = 1;
        if (cfg >= 0) {
            p = small(f, m == 1 ? 0 : 1, cfg);
            threads = cfg ? 256 : 512;
            smem = 0;
            grid = std::min(n / (cfg ? 64 : 32), occupancy.at(p) * sms);
        }
    }
    if (!p)
        p = large(w.bits, f);
    TORCH_CHECK(p, "EXL3: only 4/5-bit mul1 weights are supported");
    cuda_check(cudaLaunchCooperativeKernel(p, dim3(grid), dim3(threads), args, smem, stream));
}
void moe(MoeCoopParams p, cudaStream_t stream) {
    using namespace exl3_moe_coop_ns;
    TORCH_CHECK(sms && p.bsz >= 1 && p.bsz <= 8 && p.Hi == 2048 && p.gated && p.topk > 0,
                "EXL3 MoE: unsupported shape");
    int slots = p.bsz * p.topk;
    p.a_global = p.bsz > 1;
    auto split = [&](int base, int slices) {
        return std::max(
            1, std::min({(sms + base - 1) / base, std::max(slices / 64, 1), std::max(p.slots_max / slots, 1)}));
    };
    int ba = slots * 2 * (p.I / 128), bb = slots * (p.Ho / 128);
    p.ksplit_a = split(ba, p.Hi / 16);
    p.ksplit_b = split(bb, p.I / 16);
    void *args[] = {&p};
    if (p.a_global) {
        int items = slots * (p.Hi / 128) * 2;
        exl3_moe_coop_rot_kernel<<<(items + 15) / 16, 512, 0, stream>>>(p);
    }
    cuda_check(cudaLaunchKernel((void *)exl3_moe_coop_a_kernel<4, 2, true>, dim3(ba * p.ksplit_a), dim3(512), args,
                                smem_a_bytes<4>(p.Hi), stream));
    cuda_check(cudaLaunchKernel((void *)exl3_moe_coop_b_kernel<4, 2, true>, dim3(bb * p.ksplit_b), dim3(512), args,
                                smem_b_bytes<4>(), stream));
}
namespace {
__global__ void quant_router(const half *w, signed char *hi, signed char *lo, float *sb) {
    int row = blockIdx.x;
    float m = 0;
    __shared__ float red[8];
    for (int k = threadIdx.x; k < 2048; k += 256)
        m = fmaxf(m, fabsf(__half2float(w[row * 2048 + k])));
    for (int o = 16; o; o /= 2)
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (!(threadIdx.x & 31))
        red[threadIdx.x / 32] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < 8; ++i)
        m = fmaxf(m, red[i]);
    float scale = m > 0 ? m : 1.f, inv = __fdiv_rn(DET_QMAX, scale);
    for (int k = threadIdx.x; k < 2048; k += 256) {
        signed char h, l;
        det_quant_split(__half2float(w[row * 2048 + k]), inv, h, l);
        hi[row * 2048 + k] = h;
        lo[row * 2048 + k] = l;
    }
    if (!threadIdx.x)
        sb[row] = __fdiv_rn(scale, DET_QMAX);
}
__global__ void router_gemv(const half *x, const half *w, half *scores) {
    int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    float sum = 0;
    for (int j = lane; j < 1024; j += 32) {
        float2 a = __half22float2(((const half2 *)x)[j]);
        float2 b = __half22float2(((const half2 *)(w + row * 2048))[j]);
        sum = fmaf(a.x, b.x, sum);
        sum = fmaf(a.y, b.y, sum);
    }
    for (int o = 16; o; o /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, o);
    if (!lane)
        scores[row] = __float2half_rn(sum);
}
__global__ void router_topk(const half *scores, int64_t *sel, half *rw) {
    int lane = threadIdx.x, row = blockIdx.x;
    float keys[8];
    for (int j = 0; j < 8; ++j)
        keys[j] = __half2float(scores[row * 256 + lane + 32 * j]);
    float value = 0;
    int index = -1;
    for (int r = 0; r < 6; ++r) {
        float best = -1e30f;
        int bj = -1;
        for (int j = 0; j < 8; ++j)
            if (keys[j] > best) {
                best = keys[j];
                bj = j;
            }
        unsigned u = __float_as_uint(best);
        u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        unsigned mx = __reduce_max_sync(0xffffffff, u);
        int src = __ffs(__ballot_sync(0xffffffff, u == mx)) - 1, idx = __shfl_sync(0xffffffff, bj * 32 + lane, src);
        best = __shfl_sync(0xffffffff, best, src);
        if (idx % 32 == lane)
            keys[idx / 32] = -1e30f;
        if (lane == r) {
            float ez = exp_det(-fabsf(best));
            float base = __fdiv_rn(ez, __fadd_rn(1.f, ez));
            value = best >= 0 ? 1.f - base : base;
            index = idx;
        }
    }
    float sum = lane < 6 ? value : 0;
    for (int o = 16; o; o /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, o);
    sum = __shfl_sync(0xffffffff, sum, 0) + 1e-20f;
    if (lane < 6) {
        sel[row * 6 + lane] = index;
        rw[row * 6 + lane] = __float2half_rn(value * 128.f / sum);
    }
}
} // namespace
void prepare_router(const Router &w, cudaStream_t stream) {
    quant_router<<<256, 256, 0, stream>>>(w.weight, w.hi, w.hi + 256 * 2048, w.scales);
}
void route(const Router &w, const half *x, int rows, const RouterScratch &s, int64_t *sel, half *rw,
           cudaStream_t stream) {
    if (rows == 1)
        router_gemv<<<32, 256, 0, stream>>>(x, w.weight, s.scores);
    else {
        quant_a_kernel<<<(rows * 16 * 8 + 255) / 256, 256, 0, stream>>>(x, s.hi, s.lo, s.scales, rows, 2048, 16);
        routing_gemm_i8_kernel<<<dim3(4, 1, 8), 512, RG_SMEM, stream>>>(
            s.hi, s.lo, s.scales, w.hi, w.hi + 256 * 2048, w.scales, s.scores, s.partial, rows, 256, 2048, 16, 256);
        routing_gemm_reduce_kernel<<<(rows * 256 + 255) / 256, 256, 0, stream>>>(s.partial, s.scores, rows * 256, 8);
    }
    router_topk<<<rows, 32, 0, stream>>>(s.scores, sel, rw);
}
} // namespace fns::exl3
