// src/fa_int8.cu - prefill FlashAttention for FlintNSteel's global layers: Q·K in INT8 (mma.sync s8) and P·V in F16
// (F16 accumulation, as in ggml's MMA FA). Original code, not a transcription; the quantization is inspired by
// SageAttention (K smoothing).
//
// Opt-in (EngineOptions::fa_int8): changes the output bits.
// - Smoothed K: the per-channel mean is removed, then INT8 with one scale per head row. The mean is fixed at the first
//   batch of a sequence (p0 = 0) and kept afterwards, so all K of one call share it. Subtracting q·mean(K) from every
//   score of a row does not change the softmax; the sink is shifted by the same amount.
// - Q: INT8 with one scale per head row.
// - One block = 16 consecutive tokens x the 8 q heads of one kv head (one warp per q head, 16 mma rows = 16 tokens);
//   64-position K/V tiles in shared memory, double-buffered with cp.async.
#include "kernels.cuh"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

#define FNS_CHECK(x)                                                                                    \
    do {                                                                                                \
        cudaError_t e_ = (x);                                                                           \
        if (e_ != cudaSuccess) {                                                                        \
            std::fprintf(stderr, "CUDA %s : %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::abort();                                                                               \
        }                                                                                               \
    } while (0)

namespace fns::k {

namespace {

constexpr int D = 128, NH = 32, NKV = 4, G = 8;
constexpr int BT = 16;   // tokens per block
constexpr int KT = 64;   // K/V positions per tile
constexpr int KS = 144;  // K int8 row stride in shared memory (128 + 16: conflict-free ldmatrix)
constexpr int VS = 272;  // V half row stride (256 + 16)
constexpr int STAGE = KT * KS + KT * VS + KT * 4;  // K, V, K scales
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float sum128(float v, float * sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    v = sh[0] + sh[1] + sh[2] + sh[3];
    __syncthreads();
    return v;
}
__device__ __forceinline__ float max128(float v, float * sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    v = fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3]));
    __syncthreads();
    return v;
}

// Per-channel mean of K over rows 0 .. n - 1: grid (NKV, 16); per-slice partial sums, then atomic add.
__global__ void k_mean_k(const __half * K, float * mean, int n) {
    const int hk = blockIdx.x, i = threadIdx.x;
    float s = 0;
    for (int j = blockIdx.y; j < n; j += gridDim.y) s += __half2float(K[((size_t) j * NKV + hk) * D + i]);
    atomicAdd(&mean[hk * D + i], s / n);
}

// Per-channel equalization (SmoothQuant): q·k = Σ (q_i / s_i)(k_i s_i), with s_i = sqrt(max|Q_i| / max|K_i - mean_i|).
// This balances the ranges of both sides, so outlier channels of K or Q no longer dominate the INT8 resolution of the row.
// Statistics come from the first batch (p0 = 0): amax [2][NKV][D] (K then Q, bit patterns of positive floats, atomicMax).
__global__ void k_chan_amax(const __half * K, const float * Q, const float * mean, unsigned * amax, int n) {
    const int hk = blockIdx.x, i = threadIdx.x;
    float ka = 0, qa = 0;
    for (int j = blockIdx.y; j < n; j += gridDim.y) {
        ka = fmaxf(ka, fabsf(__half2float(K[((size_t) j * NKV + hk) * D + i]) - mean[hk * D + i]));
        for (int g = 0; g < G; ++g) qa = fmaxf(qa, fabsf(Q[((size_t) j * NH + hk * G + g) * D + i]));
    }
    atomicMax(&amax[hk * D + i], __float_as_uint(ka));
    atomicMax(&amax[NKV * D + hk * D + i], __float_as_uint(qa));
}

__global__ void k_chan_scale(const unsigned * amax, float * s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NKV * D) return;
    const float ka = __uint_as_float(amax[i]), qa = __uint_as_float(amax[NKV * D + i]);
    s[i] = ka > 0 && qa > 0 ? fminf(fmaxf(sqrtf(qa / ka), 1e-3f), 1e3f) : 1.0f;
}

// Rows lo .. hi - 1 of smoothed, equalized K -> INT8, one scale per head row: Ki [rows][NKV][D], kd [NKV][rows].
__global__ void k_quant_k(const __half * K, const float * mean, const float * cs, int8_t * Ki, float * kd, int lo,
                          int rows) {
    __shared__ float sh[4];
    const int j = lo + blockIdx.x, hk = blockIdx.y, i = threadIdx.x;
    const float y = (__half2float(K[((size_t) j * NKV + hk) * D + i]) - mean[hk * D + i]) * cs[hk * D + i];
    const float d = max128(fabsf(y), sh) / 127.0f;
    Ki[((size_t) j * NKV + hk) * D + i] = (int8_t) __float2int_rn(d > 0 ? y / d : 0.0f);
    if (i == 0) kd[(size_t) hk * rows + j] = d;
}

// Q / s -> INT8 per head row; qs = d * scale * log2(e) (scores in base 2); qc = scale * log2(e) * q·mean(K)
// (on raw Q: smoothing removes q·mean from every score of the row).
__global__ void k_quant_q(const float * Q, const float * mean, const float * cs, int8_t * Qi, float * qs, float * qc,
                          float scale) {
    __shared__ float sh[4];
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const size_t r = (size_t) t * NH + h;
    const float x = Q[r * D + i];
    const float c = sum128(x * mean[(h / G) * D + i], sh);
    const float xe = x / cs[(h / G) * D + i];
    const float d = max128(fabsf(xe), sh) / 127.0f;
    Qi[r * D + i] = (int8_t) __float2int_rn(d > 0 ? xe / d : 0.0f);
    if (i == 0) {
        qs[r] = d * scale * LOG2E;
        qc[r] = c * scale * LOG2E;
    }
}

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void cp16(void * s, const void * g) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(g));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *(const uint32_t *) &h;
}

// Queries t0 .. t0 + 15 (positions p0 + t); key j is visible if j <= p0 + t. K/V rows read: 0 .. ntile * 64 - 1,
// bounded by rows - 1 (beyond: masked). Output out [n_tok][NH][D] (ggml FA layout).
__global__ void __launch_bounds__(256, 1)
k_fa_int8(const int8_t * Qi, const float * qs, const float * qc, const int8_t * Ki, const float * kd, const __half * V,
          const float * sinks, float * out, int p0, int n_tok, int rows) {
    extern __shared__ __align__(16) uint8_t sm[];
    const int t0 = blockIdx.x * BT, hk = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q4 = lane & 3;
    const int head = hk * G + warp;
    const int tq0 = t0 + g, tq1 = t0 + g + 8;
    const size_t r0 = (size_t) min(tq0, n_tok - 1) * NH + head, r1 = (size_t) min(tq1, n_tok - 1) * NH + head;
    uint32_t qa[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        qa[ks][0] = *(const uint32_t *) (Qi + r0 * D + ks * 32 + q4 * 4);
        qa[ks][1] = *(const uint32_t *) (Qi + r1 * D + ks * 32 + q4 * 4);
        qa[ks][2] = *(const uint32_t *) (Qi + r0 * D + ks * 32 + 16 + q4 * 4);
        qa[ks][3] = *(const uint32_t *) (Qi + r1 * D + ks * 32 + 16 + q4 * 4);
    }
    const float s0 = qs[r0], s1 = qs[r1];
    const int pos0 = p0 + tq0, pos1 = p0 + tq1;
    const int ntile = (p0 + min(t0 + BT, n_tok) - 1) / KT + 1;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    uint32_t o[16][2];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = 0u;

    auto load = [&](int kt, int st) {
        uint8_t * base = sm + st * STAGE;
        const int j0 = kt * KT;
        for (int c = threadIdx.x; c < KT * 8; c += 256) {
            const int r = c >> 3, k = c & 7, j = min(j0 + r, rows - 1);
            cp16(base + r * KS + k * 16, Ki + ((size_t) j * NKV + hk) * D + k * 16);
        }
        for (int c = threadIdx.x; c < KT * 16; c += 256) {
            const int r = c >> 4, k = c & 15, j = min(j0 + r, rows - 1);
            cp16(base + KT * KS + r * VS + k * 16, (const uint8_t *) (V + ((size_t) j * NKV + hk) * D) + k * 16);
        }
        if (threadIdx.x < KT / 4) cp16(base + KT * KS + KT * VS + threadIdx.x * 16, kd + (size_t) hk * rows + j0 + threadIdx.x * 4);
    };
    load(0, 0);
    asm volatile("cp.async.commit_group;");
    for (int kt = 0; kt < ntile; ++kt) {
        const int st = kt & 1;
        if (kt + 1 < ntile) load(kt + 1, st ^ 1);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        const uint8_t * Ks = sm + st * STAGE;
        const uint8_t * Vs = Ks + KT * KS;
        const float * kds = (const float *) (Vs + KT * VS);
        int c[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0;
#pragma unroll
            for (int kp = 0; kp < 2; ++kp) {
                uint32_t b[4];
                ldsm_x4(b, Ks + (nt * 8 + (lane & 7)) * KS + kp * 64 + (lane >> 3) * 16);
                mma_s8(c[nt], qa[2 * kp], b[0], b[1]);
                mma_s8(c[nt], qa[2 * kp + 1], b[2], b[3]);
            }
        }
        const int jb = kt * KT;
        float s[8][4];
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const float2 kk = *(const float2 *) (kds + nt * 8 + q4 * 2);
            const int j = jb + nt * 8 + q4 * 2;
            s[nt][0] = j <= pos0 ? (float) c[nt][0] * s0 * kk.x : -INFINITY;
            s[nt][1] = j + 1 <= pos0 ? (float) c[nt][1] * s0 * kk.y : -INFINITY;
            s[nt][2] = j <= pos1 ? (float) c[nt][2] * s1 * kk.x : -INFINITY;
            s[nt][3] = j + 1 <= pos1 ? (float) c[nt][3] * s1 * kk.y : -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        // Row still fully masked (mx = -inf): nothing to accumulate, no NaN.
        const float a0 = mx0 == -INFINITY ? 1.0f : exp2f(m0 - mx0), a1 = mx1 == -INFINITY ? 1.0f : exp2f(m1 - mx1);
        const float b0 = mx0 == -INFINITY ? 0.0f : mx0, b1 = mx1 == -INFINITY ? 0.0f : mx1;
        m0 = mx0;
        m1 = mx1;
        l0 *= a0;
        l1 *= a1;
        uint32_t pa[4][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const float e0 = exp2f(s[nt][0] - b0), e1 = exp2f(s[nt][1] - b0);
            const float e2 = exp2f(s[nt][2] - b1), e3 = exp2f(s[nt][3] - b1);
            l0 += e0 + e1;
            l1 += e2 + e3;
            pa[nt >> 1][(nt & 1) * 2 + 0] = pack_h2(e0, e1);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_h2(e2, e3);
        }
        const __half2 h0 = __float2half2_rn(a0), h1 = __float2half2_rn(a1);
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
            *(__half2 *) &o[dn][0] = __hmul2(*(__half2 *) &o[dn][0], h0);
            *(__half2 *) &o[dn][1] = __hmul2(*(__half2 *) &o[dn][1], h1);
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int dp = 0; dp < 8; ++dp) {
                uint32_t b[4];
                ldsm_x4_t(b, Vs + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * VS + (dp * 2 + (lane >> 4)) * 16);
                mma_f16(o[2 * dp], pa[kk], b[0], b[1]);
                mma_f16(o[2 * dp + 1], pa[kk], b[2], b[3]);
            }
        }
        __syncthreads();  // the st buffer is reloaded by the next iteration
    }
    l0 += __shfl_xor_sync(0xffffffff, l0, 1);
    l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1);
    l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    // Sink shifted by the smoothing: out = Σ 2^s v / (Σ 2^s + 2^(sink·log2e - c)).
    const float sk0 = sinks[head] * LOG2E - qc[r0], sk1 = sinks[head] * LOG2E - qc[r1];
    const float M0 = fmaxf(m0, sk0), M1 = fmaxf(m1, sk1);
    const float f0 = exp2f(m0 - M0) / (l0 * exp2f(m0 - M0) + exp2f(sk0 - M0));
    const float f1 = exp2f(m1 - M1) / (l1 * exp2f(m1 - M1) + exp2f(sk1 - M1));
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
        const float2 v0 = __half22float2(*(__half2 *) &o[dn][0]);
        const float2 v1 = __half22float2(*(__half2 *) &o[dn][1]);
        if (tq0 < n_tok) *(float2 *) (out + r0 * D + dn * 8 + q4 * 2) = make_float2(v0.x * f0, v0.y * f0);
        if (tq1 < n_tok) *(float2 *) (out + r1 * D + dn * 8 + q4 * 2) = make_float2(v1.x * f1, v1.y * f1);
    }
}

// Sliding-window layers (opt-in fa_swa, changes the bits): same tiling (16 tokens x 8 q heads per block, tiles of 64
// positions); Q·K with f16 mma -> f32, P·V with f16 mma -> f16 as in ggml's FA. Only tiles holding visible positions are
// read: [pos - n_swa + 1, pos] for the 16 queries, about 9 tiles instead of the whole ring. Position p lives at slot
// p % slots; slots is a multiple of 64, so a 64-aligned tile is contiguous in the ring. The ring must hold at least
// window + n_tok - 1 slots so every visible position is present; the low end of a tile may hold a more recent position,
// which is masked by its computed position.
constexpr int KSH = 272;  // K half row stride (256 + 16)
constexpr int STAGE_H = KT * KSH + KT * VS;

__device__ __forceinline__ void mma_f16_f32(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(256, 1)
k_fa_swa(const float * Q, const __half * K, const __half * V, const float * sinks, float * out, int p0, int n_tok,
         int slots, int window, float qscale) {
    extern __shared__ __align__(16) uint8_t sm[];
    const int t0 = blockIdx.x * BT, hk = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q4 = lane & 3;
    const int head = hk * G + warp;
    const int tq0 = t0 + g, tq1 = t0 + g + 8;
    const size_t r0 = (size_t) min(tq0, n_tok - 1) * NH + head, r1 = (size_t) min(tq1, n_tok - 1) * NH + head;
    // Q * scale * log2(e) in half (ggml also multiplies Q by scale before conversion); A m16k16 fragments.
    uint32_t qa[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const float2 x0 = *(const float2 *) (Q + r0 * D + ks * 16 + q4 * 2);
        const float2 x1 = *(const float2 *) (Q + r1 * D + ks * 16 + q4 * 2);
        const float2 x2 = *(const float2 *) (Q + r0 * D + ks * 16 + 8 + q4 * 2);
        const float2 x3 = *(const float2 *) (Q + r1 * D + ks * 16 + 8 + q4 * 2);
        qa[ks][0] = pack_h2(x0.x * qscale, x0.y * qscale);
        qa[ks][1] = pack_h2(x1.x * qscale, x1.y * qscale);
        qa[ks][2] = pack_h2(x2.x * qscale, x2.y * qscale);
        qa[ks][3] = pack_h2(x3.x * qscale, x3.y * qscale);
    }
    const int pos0 = p0 + tq0, pos1 = p0 + tq1;
    const int jlo = max(0, p0 + t0 - window + 1) / KT * KT;
    const int jhi = p0 + min(t0 + BT, n_tok) - 1;
    const int ntile = (jhi - jlo) / KT + 1;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    uint32_t o[16][2];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = 0u;

    auto load = [&](int kt, int st) {
        uint8_t * base = sm + st * STAGE_H;
        const int s0 = (jlo + kt * KT) % slots;
        for (int c = threadIdx.x; c < KT * 16; c += 256) {
            const int r = c >> 4, k = c & 15;
            const size_t row = (size_t) (s0 + r) * NKV + hk;
            cp16(base + r * KSH + k * 16, (const uint8_t *) (K + row * D) + k * 16);
            cp16(base + KT * KSH + r * VS + k * 16, (const uint8_t *) (V + row * D) + k * 16);
        }
    };
    load(0, 0);
    asm volatile("cp.async.commit_group;");
    for (int kt = 0; kt < ntile; ++kt) {
        const int st = kt & 1;
        if (kt + 1 < ntile) load(kt + 1, st ^ 1);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        const uint8_t * Ks = sm + st * STAGE_H;
        const uint8_t * Vs = Ks + KT * KSH;
        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
            for (int kp = 0; kp < 4; ++kp) {
                uint32_t b[4];
                ldsm_x4(b, Ks + (nt * 8 + (lane & 7)) * KSH + kp * 64 + (lane >> 3) * 16);
                mma_f16_f32(s[nt], qa[2 * kp], b[0], b[1]);
                mma_f16_f32(s[nt], qa[2 * kp + 1], b[2], b[3]);
            }
        }
        const int jb = jlo + kt * KT;
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int j = jb + nt * 8 + q4 * 2;
            if (!(j <= pos0 && pos0 - j < window)) s[nt][0] = -INFINITY;
            if (!(j + 1 <= pos0 && pos0 - j - 1 < window)) s[nt][1] = -INFINITY;
            if (!(j <= pos1 && pos1 - j < window)) s[nt][2] = -INFINITY;
            if (!(j + 1 <= pos1 && pos1 - j - 1 < window)) s[nt][3] = -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        const float a0 = mx0 == -INFINITY ? 1.0f : exp2f(m0 - mx0), a1 = mx1 == -INFINITY ? 1.0f : exp2f(m1 - mx1);
        const float b0 = mx0 == -INFINITY ? 0.0f : mx0, b1 = mx1 == -INFINITY ? 0.0f : mx1;
        m0 = mx0;
        m1 = mx1;
        l0 *= a0;
        l1 *= a1;
        uint32_t pa[4][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const float e0 = exp2f(s[nt][0] - b0), e1 = exp2f(s[nt][1] - b0);
            const float e2 = exp2f(s[nt][2] - b1), e3 = exp2f(s[nt][3] - b1);
            l0 += e0 + e1;
            l1 += e2 + e3;
            pa[nt >> 1][(nt & 1) * 2 + 0] = pack_h2(e0, e1);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_h2(e2, e3);
        }
        const __half2 h0 = __float2half2_rn(a0), h1 = __float2half2_rn(a1);
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
            *(__half2 *) &o[dn][0] = __hmul2(*(__half2 *) &o[dn][0], h0);
            *(__half2 *) &o[dn][1] = __hmul2(*(__half2 *) &o[dn][1], h1);
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int dp = 0; dp < 8; ++dp) {
                uint32_t b[4];
                ldsm_x4_t(b, Vs + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * VS + (dp * 2 + (lane >> 4)) * 16);
                mma_f16(o[2 * dp], pa[kk], b[0], b[1]);
                mma_f16(o[2 * dp + 1], pa[kk], b[2], b[3]);
            }
        }
        __syncthreads();
    }
    l0 += __shfl_xor_sync(0xffffffff, l0, 1);
    l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1);
    l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    const float sk = sinks[head] * LOG2E;
    const float M0 = fmaxf(m0, sk), M1 = fmaxf(m1, sk);
    const float f0 = exp2f(m0 - M0) / (l0 * exp2f(m0 - M0) + exp2f(sk - M0));
    const float f1 = exp2f(m1 - M1) / (l1 * exp2f(m1 - M1) + exp2f(sk - M1));
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
        const float2 v0 = __half22float2(*(__half2 *) &o[dn][0]);
        const float2 v1 = __half22float2(*(__half2 *) &o[dn][1]);
        if (tq0 < n_tok) *(float2 *) (out + r0 * D + dn * 8 + q4 * 2) = make_float2(v0.x * f0, v0.y * f0);
        if (tq1 < n_tok) *(float2 *) (out + r1 * D + dn * 8 + q4 * 2) = make_float2(v1.x * f1, v1.y * f1);
    }
}

}  // namespace

void fa_swa_prefill(const float * q, const void * kc, const void * vc, const float * sinks, float * out, int n_tok,
                    int p0, int slots, int window, float scale, cudaStream_t st) {
    if (slots % KT || slots < window + n_tok - 1) {
        std::fprintf(stderr, "fa_swa_prefill: invalid ring of %d slots (multiple of %d, >= %d)\n", slots, KT,
                     window + n_tok - 1);
        std::abort();
    }
    static bool attr = false;
    if (!attr) {
        FNS_CHECK(cudaFuncSetAttribute(k_fa_swa, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * STAGE_H));
        attr = true;
    }
    k_fa_swa<<<dim3((n_tok + BT - 1) / BT, NKV), 256, 2 * STAGE_H, st>>>(q, (const __half *) kc, (const __half *) vc,
                                                                         sinks, out, p0, n_tok, slots, window,
                                                                         scale * LOG2E);
    FNS_CHECK(cudaGetLastError());
}

size_t fa_int8_q_bytes(int n_tok) { return (size_t) (n_tok + BT) * NH * (D + 8); }
size_t fa_int8_k_bytes(int rows) { return (size_t) (rows + KT) * NKV * (D + 4); }
size_t fa_int8_stat_floats() { return (size_t) 4 * NKV * D; }

void fa_int8_prefill(const float * q, const void * kc, const void * vc, const float * sinks, float * out, int n_tok,
                     int p0, int rows, float scale, FaInt8State & s, cudaStream_t st) {
    if (rows % KT) {
        std::fprintf(stderr, "fa_int8_prefill: cache rows must be a multiple of %d\n", KT);
        std::abort();
    }
    int8_t * qi = (int8_t *) s.q;
    float * qs = (float *) (qi + (size_t) (n_tok + BT) * NH * D);
    float * qc = qs + (size_t) (n_tok + BT) * NH;
    int8_t * ki = (int8_t *) s.k;
    float * kd = (float *) (ki + (size_t) (rows + KT) * NKV * D);
    const __half * K = (const __half *) kc;
    // s.mean: [mean | cs | amax K | amax Q], 4 × NKV × D floats (fa_int8_stat_floats). Statistics are fixed at the first
    // batch of the sequence; rows to (re)quantize are those written since the last call.
    float * cs = s.mean + NKV * D;
    unsigned * amax = (unsigned *) (s.mean + 2 * NKV * D);
    if (p0 == 0) {
        FNS_CHECK(cudaMemsetAsync(s.mean, 0, 4 * NKV * D * sizeof(float), st));
        k_mean_k<<<dim3(NKV, 16), D, 0, st>>>(K, s.mean, n_tok);
        k_chan_amax<<<dim3(NKV, 16), D, 0, st>>>(K, q, s.mean, amax, n_tok);
        k_chan_scale<<<(NKV * D + 127) / 128, 128, 0, st>>>(amax, cs);
        FNS_CHECK(cudaGetLastError());
    }
    const int lo = p0 == 0 ? 0 : std::min(s.quantized, p0), hi = p0 + n_tok;
    k_quant_k<<<dim3(hi - lo, NKV), D, 0, st>>>(K, s.mean, cs, ki, kd, lo, rows);
    FNS_CHECK(cudaGetLastError());
    s.quantized = hi;
    k_quant_q<<<dim3(n_tok, NH), D, 0, st>>>(q, s.mean, cs, qi, qs, qc, scale);
    FNS_CHECK(cudaGetLastError());
    static bool attr = false;
    if (!attr) {
        FNS_CHECK(cudaFuncSetAttribute(k_fa_int8, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * STAGE));
        attr = true;
    }
    k_fa_int8<<<dim3((n_tok + BT - 1) / BT, NKV), 256, 2 * STAGE, st>>>(qi, qs, qc, ki, kd, (const __half *) vc, sinks,
                                                                       out, p0, n_tok, rows);
    FNS_CHECK(cudaGetLastError());
}

}  // namespace fns::k
