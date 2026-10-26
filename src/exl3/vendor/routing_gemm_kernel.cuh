/*
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Origin: exllamav3 v1.6.2 (4c842731), patch 3b013de9, exllamav3_ext/routing_gemm.cu.
Modifications: Tensor frontend removed, arithmetic unchanged.
*/
#pragma once
#include "det_gemm.cuh"
#define RG_BM 128
#define RG_BN 64
#define RG_THREADS 512
// One stage on ROCm: its copies are synchronous (rocm/det_gemm_rocm.cuh), so a second stage would overlap
// nothing, and two do not fit RDNA's 64 KB of LDS per workgroup
#if defined(USE_ROCM)
    #define RG_STAGES 1
#else
    #define RG_STAGES 2
#endif
#define RG_KCH 128
#define RG_A_BYTES (2 * RG_BM * RG_KCH)
#define RG_B_BYTES (2 * RG_BN * RG_KCH)
#define RG_S_BYTES (RG_BM * 4)
#define RG_STAGE_BYTES (RG_A_BYTES + RG_B_BYTES + RG_S_BYTES)
#define RG_SMEM (RG_STAGES * RG_STAGE_BYTES)
#define RG_STATIC_ROWS 64

// A (R, K) half -> hi/lo (R, K) int8, sa (R, KC) fp32. One thread per 16 k of one row; the 8
// threads of a 128-chunk share the max by shuffle
__global__ __launch_bounds__(256)
void quant_a_kernel(const half* __restrict__ a, signed char* __restrict__ ahi, signed char* __restrict__ alo,
                    float* __restrict__ sa, const int R, const int K, const int KC)
{
    const int idx = blockIdx.x * 256 + threadIdx.x;
    const int eighth = idx & 7, chunk = (idx >> 3) % KC, row = (idx >> 3) / KC;
    if (row >= R) return;
    const int k = chunk * RG_KCH + eighth * 16;
    float v[16];
    #pragma unroll
    for (int q = 0; q < 2; ++q)
    {
        int4 raw = make_int4(0, 0, 0, 0);
        if (k + q * 8 < K) raw = *(const int4*) (a + (size_t) row * K + k + q * 8);
        const half2* h2 = (const half2*) &raw;
        #pragma unroll
        for (int u = 0; u < 4; ++u) { float2 f = __half22float2(h2[u]); v[q * 8 + u * 2] = f.x; v[q * 8 + u * 2 + 1] = f.y; }
    }
    float m = 0.0f;
    #pragma unroll
    for (int q = 0; q < 16; ++q) m = fmaxf(m, fabsf(v[q]));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
    const float scale = m > 0.0f ? m : 1.0f;
    int4 hi4, lo4;
    det_quant16(v, __fdiv_rn(DET_QMAX, scale), hi4, lo4);
    if (k < K)
    {
        *(int4*) (ahi + (size_t) row * K + k) = hi4;
        *(int4*) (alo + (size_t) row * K + k) = lo4;
    }
    if (eighth == 0) sa[(size_t) row * KC + chunk] = __fdiv_rn(scale, DET_QMAX);
}

__global__ __launch_bounds__(RG_THREADS)
void routing_gemm_i8_kernel
(
    const signed char* __restrict__ ahi, const signed char* __restrict__ alo, const float* __restrict__ sa,   // (R, K), (R, K), (R, KC)
    const signed char* __restrict__ bhi, const signed char* __restrict__ blo, const float* __restrict__ sb,   // (E, K), (E, K), (E)
    half* __restrict__ c, float* __restrict__ part,                                                           // (R, E) half, or (S, R, E) fp32 partials
    const int R, const int E, const int K, const int KC, const int kslice
)
{
    extern __shared__ __align__(128) unsigned char dsm[];
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const int wm = warp / 4, wn = warp % 4;
    const int g = lane / 4, tg = lane % 4;
    const int r0 = blockIdx.y * RG_BM, e0 = blockIdx.x * RG_BN;
    const int k_beg = blockIdx.z * kslice, k_end = min(k_beg + kslice, K);
    const int n_chunks = (k_end - k_beg + RG_KCH - 1) / RG_KCH;

    auto st_ahi = [&](int st) { return dsm + st * RG_STAGE_BYTES; };
    auto st_alo = [&](int st) { return dsm + st * RG_STAGE_BYTES + RG_BM * RG_KCH; };
    auto st_bhi = [&](int st) { return dsm + st * RG_STAGE_BYTES + RG_A_BYTES; };
    auto st_blo = [&](int st) { return dsm + st * RG_STAGE_BYTES + RG_A_BYTES + RG_BN * RG_KCH; };
    auto st_sa  = [&](int st) { return (float*) (dsm + st * RG_STAGE_BYTES + RG_A_BYTES + RG_B_BYTES); };

    auto issue = [&](int chunk, int st)
    {
        const int k0 = k_beg + chunk * RG_KCH;
        #pragma unroll
        for (int q = 0; q < 2; ++q)
        {
            const int i = t + q * RG_THREADS, row = i / 8, cc = i % 8;
            const int gr = r0 + row, k = k0 + cc * 16;
            const int bytes = (gr < R) ? max(0, min(16, k_end - k)) : 0;
            const size_t off = bytes ? (size_t) gr * K + k : 0;
            det_cp_async16(det_smem_u32(st_ahi(st) + det_swz8(row, cc)), ahi + off, bytes);
            det_cp_async16(det_smem_u32(st_alo(st) + det_swz8(row, cc)), alo + off, bytes);
        }
        {
            const int row = t / 8, cc = t % 8;
            const int ge = e0 + row, k = k0 + cc * 16;
            const int bytes = (ge < E) ? max(0, min(16, k_end - k)) : 0;
            const size_t off = bytes ? (size_t) ge * K + k : 0;
            det_cp_async16(det_smem_u32(st_bhi(st) + det_swz8(row, cc)), bhi + off, bytes);
            det_cp_async16(det_smem_u32(st_blo(st) + det_swz8(row, cc)), blo + off, bytes);
        }
        if (t < RG_BM / 4)
        {
            float v4[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u)
            {
                const int gr = r0 + t * 4 + u;
                v4[u] = (gr < R) ? sa[(size_t) gr * KC + (k0 / RG_KCH)] : 0.0f;
            }
            *(float4*) (st_sa(st) + t * 4) = make_float4(v4[0], v4[1], v4[2], v4[3]);
        }
    };

#if defined(USE_ROCM)
    DetWmmaTile<2, 1> wt;
    wt.clear();
#else
    float facc[2][2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q) facc[i][j][q] = 0.0f;
#endif

    issue(0, 0); det_cp_async_commit();
    for (int chunk = 0; chunk < n_chunks; ++chunk)
    {
        det_cp_async_wait<0>();
        __syncthreads();
        if (RG_STAGES > 1 && chunk + 1 < n_chunks) issue(chunk + 1, (chunk + 1) % RG_STAGES);
        det_cp_async_commit();
        const int st = chunk % RG_STAGES;
        const unsigned a_hi = det_smem_u32(st_ahi(st)), a_lo = det_smem_u32(st_alo(st));
        const unsigned b_hi = det_smem_u32(st_bhi(st)), b_lo = det_smem_u32(st_blo(st));
        const float* sa_st = st_sa(st);

#if defined(USE_ROCM)
        wt.chunk<128, RG_KCH>(a_hi, a_lo, wm * 32, b_hi, b_lo, wn * 16, sa_st, lane);
#else
        int acc_hh[2][2][4], acc_x[2][2][4];
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) { acc_hh[i][j][q] = 0; acc_x[i][j][q] = 0; }

        #pragma unroll
        for (int ks = 0; ks < RG_KCH; ks += 32)
        {
            unsigned ah[2][4], al[2][4], bh[2][2], bl[2][2], r4[4];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                det_load_a(ah[i], a_hi, wm * 32 + i * 16, ks / 16, lane);
                det_load_a(al[i], a_lo, wm * 32 + i * 16, ks / 16, lane);
            }
            det_load_b2(r4, b_hi, wn * 16, ks / 16, lane);
            bh[0][0] = r4[0]; bh[0][1] = r4[1]; bh[1][0] = r4[2]; bh[1][1] = r4[3];
            det_load_b2(r4, b_lo, wn * 16, ks / 16, lane);
            bl[0][0] = r4[0]; bl[0][1] = r4[1]; bl[1][0] = r4[2]; bl[1][1] = r4[3];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    det_mma3(acc_hh[i][j], acc_x[i][j], ah[i], al[i], bh[j], bl[j]);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const float s0 = sa_st[wm * 32 + i * 16 + g], s1 = sa_st[wm * 32 + i * 16 + g + 8];
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q)
                    facc[i][j][q] = det_flush(acc_hh[i][j][q], acc_x[i][j][q], q >= 2 ? s1 : s0, facc[i][j][q]);
        }
#endif
        // Single buffer: the next chunk can only land once everyone is done with this one
        if (RG_STAGES == 1 && chunk + 1 < n_chunks)
        {
            __syncthreads();
            issue(chunk + 1, 0);
        }
    }
    det_cp_async_wait<0>();
    __syncthreads();
    float (*cs_)[RG_BN + 4] = (float (*)[RG_BN + 4]) dsm;
#if defined(USE_ROCM)
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int v = 0; v < 8; ++v)
            cs_[wm * 32 + i * 16 + det_wmma_row(v, lane)][wn * 16 + (lane & 15)] = wt.acc[i][0][v];
#else
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q)
                cs_[wm * 32 + i * 16 + g + (q >= 2 ? 8 : 0)][wn * 16 + j * 8 + tg * 2 + (q & 1)] = facc[i][j][q];
#endif
    __syncthreads();
    if (part) part += (size_t) blockIdx.z * R * E;
    #pragma unroll
    for (int q = 0; q < (RG_BM * RG_BN) / RG_THREADS; ++q)
    {
        const int i = (t + q * RG_THREADS) / RG_BN, j = (t + q * RG_THREADS) % RG_BN;
        if (r0 + i < R && e0 + j < E)
        {
            float v = __fmul_rn(cs_[i][j], sb[e0 + j]);
            if (part) part[(size_t) (r0 + i) * E + e0 + j] = v;
            else c[(size_t) (r0 + i) * E + e0 + j] = __float2half_rn(v);
        }
    }
}

__global__ __launch_bounds__(256)
void routing_gemm_reduce_kernel(const float* __restrict__ part, half* __restrict__ c, const int n, const int S)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
    for (int s = 0; s < S; ++s) v = __fadd_rn(v, part[(size_t) s * n + i]);
    c[i] = __float2half_rn(v);
}

// Split-K slices: a function of the shape only
static int rg_slices(int R, int E, int KC)
{
    const int tiles = CEIL_DIVIDE(E, RG_BN) * CEIL_DIVIDE(R, RG_BM);
    int S = 1;
    while (tiles * S < 256 && S < 8 && S * 2 <= KC) S *= 2;
    return S;
}

