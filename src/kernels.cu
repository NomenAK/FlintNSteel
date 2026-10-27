// src/kernels.cu - basic CUDA kernels of the engine (see kernels.cuh for the ggml references).
#include "kernels.cuh"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define FNS_CHECK(x)                                                                                  \
    do {                                                                                              \
        cudaError_t e_ = (x);                                                                         \
        if (e_ != cudaSuccess) {                                                                      \
            std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));      \
            std::abort();                                                                             \
        }                                                                                             \
    } while (0)

namespace fns::k {

namespace {

constexpr int kBlock = 256;

template <bool Aligned>
__global__ void k_swap_bytes(uint8_t * a, uint8_t * b, size_t n) {
    const size_t fragments = n / 16 + (n % 16 != 0);
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < fragments;
         i += (size_t) blockDim.x * gridDim.x) {
        const size_t offset = i * 16;
        if (Aligned && n - offset >= 16) {
            uint4 * pa = reinterpret_cast<uint4 *>(a + offset);
            uint4 * pb = reinterpret_cast<uint4 *>(b + offset);
            const uint4 va = *pa, vb = *pb;
            *pa = vb;
            *pb = va;
        } else {
            const size_t count = n - offset < 16 ? n - offset : 16;
            for (size_t j = 0; j < count; ++j) {
                const uint8_t va = a[offset + j], vb = b[offset + j];
                a[offset + j] = vb;
                b[offset + j] = va;
            }
        }
    }
}

// block_q8_1 from ggml-common.h (MIT, The ggml authors): 32 int8 values and (d, Σx) as half2. Same layout, checked by the
// static_assert; quantize_row_q8_1_cuda and ggml's MMVQ read these blocks.
struct block_q8_1 {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 of ggml: 36 bytes");

// block_q8_0 from ggml-common.h (MIT, The ggml authors): F16 scale, then 32 int8 values. Used for the 8-bit global KV cache:
// one head row (128 components) is 4 blocks; the cache is [slots][n_head_kv][4] blocks, same order as the F16 cache.
struct block_q8_0 {
    __half d;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 of ggml: 34 bytes");

// One warp quantizes a block of 32 values, one per lane. Follows quantize_f32_q8_0_block in ggml-cuda (cpy-utils.cuh:137-154,
// llama.cpp 19ab5a5e): d = amax / 127, q = roundf(x / d); d = 0 gives q = 0.
__device__ __forceinline__ void store_q8_0(block_q8_0 * b, float x, int lane) {
    float amax = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    b->qs[lane] = (int8_t) roundf(x * id);
    if (lane == 0) b->d = __float2half(d);
}

__device__ float block_sum(float v, float * sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = lane < nw ? sh[lane] : 0.0f;
    if (warp == 0) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}

__device__ float block_max(float v, float * sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = lane < nw ? sh[lane] : -FLT_MAX;
    if (warp == 0) {
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}

// Row token[blockIdx.y] of token_embd (Q8_0 on the GPU) -> out[blockIdx.y] F32. Follows ggml dequantize_row_q8_0.
__global__ void k_embed_q8_0(const uint8_t * table, const int32_t * token, float * out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t * blk = table + (size_t) token[blockIdx.y] * (n / 32) * 34 + (i / 32) * 34;
    const float d = __half2float(*(const __half *) blk);
    out[(size_t) blockIdx.y * n + i] = d * (float) (int8_t) blk[2 + (i % 32)];
}

__global__ void k_rms_norm_mul(const float * x, const float * w, float * out, int n, float eps) {
    __shared__ float sh[32];
    const float * xr = x + (size_t) blockIdx.x * n;
    float * orow = out + (size_t) blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float scale = rsqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) orow[i] = scale * xr[i] * w[i];
}

// RMSNorm of one row of n = 2048 (1024 threads, 2 values per thread), followed by ggml's quantize_q8_1 (quantize.cu:53-101,
// same arithmetic: d = amax/127, q = roundf(x/d), ds = (d, Σx) in half) on the output. Also writes the F32 output, read
// by the BF16 router and the traces. The 32 values of a Q8_1 block sit on the 32 lanes of one warp, as in ggml: one pass,
// no round trip through global memory. One block per row.
__global__ void __launch_bounds__(1024) k_rms_norm_q8_1(const float * x, const float * w, float * out, void * xq,
                                                       int n, float eps) {
    __shared__ float sh[32];
    x += (size_t) blockIdx.x * n;
    out += (size_t) blockIdx.x * n;
    float v[2];
    float s = 0.0f;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        v[r] = x[r * 1024 + threadIdx.x];
        s += v[r] * v[r];
    }
    s = block_sum(s, sh);
    const float scale = rsqrtf(s / n + eps);
    block_q8_1 * y = (block_q8_1 *) xq + (size_t) blockIdx.x * (n / 32);  // n = 2048: row with no padding
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int i = r * 1024 + threadIdx.x;
        const float xi = scale * v[r] * w[i];
        out[i] = xi;
        float amax = fabsf(xi), sum = xi;
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
            sum += __shfl_xor_sync(0xffffffff, sum, o);
        }
        const float d = amax / 127.0f;
        y[i / 32].qs[i % 32] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
        if (i % 32 == 0) y[i / 32].ds = make_half2(d, sum);
    }
}
// NEOX RoPE: the pair (i, i + head_dim/2) rotates by theta_i = pos * theta_scale^i, i < head_dim/2.
__global__ void k_rope_neox(float * x, int head_dim, const int32_t * pos_p, float theta_scale) {
    const int i = threadIdx.x;
    const int half = head_dim / 2;
    if (i >= half) return;
    float * h = x + (size_t) blockIdx.x * head_dim;
    const float theta = (float) *pos_p * powf(theta_scale, (2 * i) / 2.0f);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = h[i], x1 = h[i + half];
    h[i] = x0 * c - x1 * s;
    h[i + half] = x0 * s + x1 * c;
}

__global__ void k_store_kv(const float * k, const float * v, __half * kc, __half * vc, const int32_t * pos_p,
                           int slots, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int slot = *pos_p % slots;
    kc[(size_t) slot * n + i] = __float2half(k[i]);
    vc[(size_t) slot * n + i] = __float2half(v[i]);
}

// After the QKV GEMV: per-head RMSNorm of Q and K (weights qn, kn), NEOX RoPE if `rope`, then K and V as half into the cache
// at slot pos % slots. Grid: (n_head + n_head_kv, cols) blocks of 128 threads (one head of one column per block, one
// component per thread); column c reads q, k, v at + c * stride and uses position *pos_p + c. Same operations, same
// order as rms_norm_mul + rope_neox + store_kv.
__global__ void __launch_bounds__(128) k_qk_norm_rope_store(float * q, float * k, const float * v, const float * qn,
                                                            const float * kn, __half * kc, __half * vc,
                                                            const int32_t * pos_p, int n_head, int n_head_kv,
                                                            int slots, float eps, float theta_scale, int rope,
                                                            int stride, bool q8) {
    constexpr int D = 128;
    __shared__ float sh[32];
    __shared__ float row[D];
    const size_t col = (size_t) blockIdx.y * stride;
    const bool is_q = blockIdx.x < (unsigned) n_head;
    const int head = is_q ? blockIdx.x : blockIdx.x - n_head;
    float * x = (is_q ? q : k) + col + (size_t) head * D;
    const float * w = is_q ? qn : kn;
    const int i = threadIdx.x;
    const float xi = x[i];
    const float s = block_sum(xi * xi, sh);
    const float scale = rsqrtf(s / D + eps);
    row[i] = scale * xi * w[i];
    __syncthreads();
    const int pos = *pos_p + blockIdx.y;
    float y = row[i];
    if (rope) {
        const int j = i % (D / 2);
        const float theta = (float) pos * powf(theta_scale, (2 * j) / 2.0f);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = row[j], x1 = row[j + D / 2];
        y = i < D / 2 ? x0 * c - x1 * sn : x0 * sn + x1 * c;
    }
    x[i] = y;
    if (!is_q) {
        const size_t row = (size_t) (pos % slots) * n_head_kv + head;
        if (q8) {
            const float vi = v[col + (size_t) head * D + i];
            store_q8_0((block_q8_0 *) kc + row * (D / 32) + i / 32, y, i & 31);
            store_q8_0((block_q8_0 *) vc + row * (D / 32) + i / 32, vi, i & 31);
        } else {
            kc[row * D + i] = __float2half(y);
            vc[row * D + i] = __float2half(v[col + (size_t) head * D + i]);
        }
    }
}

// Split-K attention for one token. Grid (n_chunk, n_head_kv, cols): a block reads one chunk of kChunk cache positions for the
// G = n_head / n_head_kv query heads of one KV head (GQA: K and V are read once for G heads). One warp per position; each
// lane holds 4 of the head_dim (= 128) components. Partial results (m, s, Σ p v) per head and chunk are merged by
// k_attention_reduce with the sink. Scores never go through global memory. Columns (W-column verification): blockIdx.x =
// chunk * cols + column, so the W blocks reading the same cache chunk are adjacent in scheduling order (L2 re-reads, not
// DRAM). Column z: q + z * q_stride, position *pos_p + z, partials offset by z * part_stride floats. Each block recomputes
// exactly the single-token computation: same bits.
constexpr int kChunk = 64;  // positions per block: 36 blocks for a sliding layer, 2048 at 32K context (82 SMs)
constexpr int kAttWarps = 8;
constexpr int kG = 8;  // q heads per KV head (32 / 4): constant so that q, m, s, o stay in registers

// One step of the transposed butterfly: N values per lane -> N/2; the upper half of the warp (bit OFF) keeps the upper half
// of the array. Compile-time indices so that v stays in registers.
template <int OFF, int N> __device__ __forceinline__ void butterfly_step(float (&v)[32], int lane) {
    const bool up = lane & OFF;
#pragma unroll
    for (int k = 0; k < N / 2; ++k) {
        const float send = up ? v[k] : v[k + N / 2];
        const float keep = up ? v[k + N / 2] : v[k];
        v[k] = keep + __shfl_xor_sync(0xffffffff, send, OFF);
    }
}

template <bool Q8>
__global__ void __launch_bounds__(kAttWarps * 32)
k_attention_part(const float * q, const void * kc_, const void * vc_, float * part_o, float2 * part_ms,
                 int n_head_kv, const int32_t * pos_p, int window, int slots, float scale, int q_stride,
                 int part_stride, int cols, int n_chunk) {
    constexpr int D = 128, G = kG;
    const int hk = blockIdx.y;
    const int col = blockIdx.x % cols, chunk = blockIdx.x / cols;
    const int pos = *pos_p + col;
    q += (size_t) col * q_stride;
    part_o += (size_t) col * part_stride;
    part_ms = (float2 *) ((float *) part_ms + (size_t) col * part_stride);
    const int lo = window > 0 ? max(0, pos - window + 1) : 0;
    const int j0 = lo + chunk * kChunk;
    const int j1 = min(j0 + kChunk, pos + 1);
    // Chunk past `pos`: nothing to read. The reducer only reads useful chunks (same n_used computation).
    if (j0 > pos) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int kv_stride = n_head_kv * D;
    const __half * kc = (const __half *) kc_;
    const __half * vc = (const __half *) vc_;
    // Q8_0: the lane reads its 4 components (lane * 4 .. + 3) from block lane / 8 of the head row.
    const block_q8_0 * kq = (const block_q8_0 *) kc_;
    const block_q8_0 * vq = (const block_q8_0 *) vc_;

    __shared__ float s_m[kAttWarps][G], s_s[kAttWarps][G];
    __shared__ float s_o[kAttWarps][G][D];

    float qv[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int c = 0; c < 4; ++c) qv[g][c] = q[(size_t) (hk * G + g) * D + lane * 4 + c] * scale;

    // Online softmax state: lane l tracks head l/4 (m and s replicated across the 4 lanes of the group); o per head.
    float m = -FLT_MAX, s = 0.0f, o[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int c = 0; c < 4; ++c) o[g][c] = 0.0f;
    // Four positions per iteration and per warp (stride kAttWarps). The 32 partial products (8 heads x 4 positions) are reduced
    // by a transposed butterfly (31 shuffles instead of 160): lane l gets the full score of head l/4 at position l%4. One expf
    // for the score and one for the rescale per lane and iteration, instead of 16 per position.
    constexpr int T = 4;
    const int js = j0 % slots;
    const int ig = lane & 3;
    for (int j = j0 + warp; j < j1; j += T * kAttWarps) {
        uint2 kr[T], vr[T];
        float ks[T], vs[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const int jt = min(j + t * kAttWarps, j1 - 1);  // beyond the chunk: a valid position is re-read and masked
            const int r = js + (jt - j0);
            const size_t slot = (size_t) (r < slots ? r : r - slots);
            if constexpr (Q8) {
                // 34-byte block: qs is only 2-byte aligned, hence two 16-bit reads.
                const size_t bi = (slot * n_head_kv + hk) * (D / 32) + lane / 8;
                const uint16_t * kq16 = (const uint16_t *) (kq[bi].qs + (lane % 8) * 4);
                const uint16_t * vq16 = (const uint16_t *) (vq[bi].qs + (lane % 8) * 4);
                kr[t].x = (uint32_t) kq16[0] | ((uint32_t) kq16[1] << 16);
                vr[t].x = (uint32_t) vq16[0] | ((uint32_t) vq16[1] << 16);
                ks[t] = __half2float(kq[bi].d);
                vs[t] = __half2float(vq[bi].d);
            } else {
                const size_t b = slot * kv_stride + (size_t) hk * D + lane * 4;
                kr[t] = *(const uint2 *) (kc + b);
                vr[t] = *(const uint2 *) (vc + b);
            }
        }
        static_assert(G * T == 32, "butterfly written for 32 products per lane");
        float v[32];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float2 k01, k23;
            if constexpr (Q8) {
                // Component = d * q (ggml dequantize_q8_0), then the same product as F16.
                const char4 c = *(const char4 *) &kr[t].x;
                k01 = make_float2(ks[t] * (float) c.x, ks[t] * (float) c.y);
                k23 = make_float2(ks[t] * (float) c.z, ks[t] * (float) c.w);
            } else {
                k01 = __half22float2(*(const __half2 *) &kr[t].x);
                k23 = __half22float2(*(const __half2 *) &kr[t].y);
            }
#pragma unroll
            for (int g = 0; g < G; ++g)
                v[g * T + t] = qv[g][0] * k01.x + qv[g][1] * k01.y + qv[g][2] * k23.x + qv[g][3] * k23.y;
        }
        butterfly_step<16, 32>(v, lane);
        butterfly_step<8, 16>(v, lane);
        butterfly_step<4, 8>(v, lane);
        butterfly_step<2, 4>(v, lane);
        butterfly_step<1, 2>(v, lane);
        const bool ok = j + ig * kAttWarps < j1;
        float mx = ok ? v[0] : -FLT_MAX;
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        const float mn = fmaxf(m, mx);
        const float a = expf(m - mn), p = ok ? expf(v[0] - mn) : 0.0f;
        float ps = p + __shfl_xor_sync(0xffffffff, p, 1);
        ps += __shfl_xor_sync(0xffffffff, ps, 2);
        s = s * a + ps;
        m = mn;
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float ag = __shfl_sync(0xffffffff, a, 4 * g);
#pragma unroll
            for (int c = 0; c < 4; ++c) o[g][c] *= ag;
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float2 v01, v23;
            if constexpr (Q8) {
                const char4 c = *(const char4 *) &vr[t].x;
                v01 = make_float2(vs[t] * (float) c.x, vs[t] * (float) c.y);
                v23 = make_float2(vs[t] * (float) c.z, vs[t] * (float) c.w);
            } else {
                v01 = __half22float2(*(const __half2 *) &vr[t].x);
                v23 = __half22float2(*(const __half2 *) &vr[t].y);
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float pg = __shfl_sync(0xffffffff, p, 4 * g + t);
                o[g][0] += pg * v01.x;
                o[g][1] += pg * v01.y;
                o[g][2] += pg * v23.x;
                o[g][3] += pg * v23.y;
            }
        }
    }
    if (ig == 0) {
        s_m[warp][lane >> 2] = m;
        s_s[warp][lane >> 2] = s;
    }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int c = 0; c < 4; ++c) s_o[warp][g][lane * 4 + c] = o[g][c];
    __syncthreads();
    // Warp merge: warp g merges head g.
    if (warp < G) {
        const int g = warp;
        float mm = -FLT_MAX;
        for (int w = 0; w < kAttWarps; ++w) mm = fmaxf(mm, s_m[w][g]);
        float ss = 0.0f, oo[4] = {0, 0, 0, 0};
        for (int w = 0; w < kAttWarps; ++w) {
            const float a = s_m[w][g] == -FLT_MAX ? 0.0f : expf(s_m[w][g] - mm);
            ss += s_s[w][g] * a;
            for (int c = 0; c < 4; ++c) oo[c] += s_o[w][g][lane * 4 + c] * a;
        }
        const int h = hk * G + g;
        const size_t pi = (size_t) h * n_chunk + chunk;
        for (int c = 0; c < 4; ++c) part_o[pi * D + lane * 4 + c] = oo[c];
        if (lane == 0) part_ms[pi] = make_float2(mm, ss);
    }
}

// Merges each head's chunks and adds the sink: out = Σ_c e^(m_c - M) o_c / (Σ_c e^(m_c - M) s_c + e^(sink - M)).
// One block of 512 threads per head and column (grid (n_head, cols)): 4 groups of 128 share the chunks (c ≡ group mod 4),
// then combine.
__global__ void __launch_bounds__(512) k_attention_reduce(const float * part_o, const float2 * part_ms,
                                                          const float * sinks, float * out, int n_chunk,
                                                          const int32_t * pos_p, int window, int part_stride) {
    constexpr int D = 128, NG = 4;
    __shared__ float s_m[NG], s_S[NG], s_O[NG][D];
    const int h = blockIdx.x, d = threadIdx.x % D, grp = threadIdx.x / D;
    const int pos = *pos_p + blockIdx.y;
    part_o += (size_t) blockIdx.y * part_stride;
    part_ms = (const float2 *) ((const float *) part_ms + (size_t) blockIdx.y * part_stride);
    out += (size_t) blockIdx.y * gridDim.x * D;
    const int lo = window > 0 ? max(0, pos - window + 1) : 0;
    const int n_used = min(n_chunk, (pos - lo) / kChunk + 1);  // chunks written by k_attention_part
    const float2 * ms = part_ms + (size_t) h * n_chunk;
    float M = -FLT_MAX;
    for (int c = grp; c < n_used; c += NG) M = fmaxf(M, ms[c].x);
    float S = 0.0f, O = 0.0f;
    for (int c = grp; c < n_used; c += NG) {
        const float a = expf(ms[c].x - M);
        S += ms[c].y * a;
        O += part_o[((size_t) h * n_chunk + c) * D + d] * a;
    }
    if (d == 0) {
        s_m[grp] = M;
        s_S[grp] = S;
    }
    s_O[grp][d] = O;
    __syncthreads();
    if (grp != 0) return;
    const float sink = sinks[h];
    float MM = sink;
    for (int g = 0; g < NG; ++g) MM = fmaxf(MM, s_m[g]);
    float SS = expf(sink - MM), OO = 0.0f;
    for (int g = 0; g < NG; ++g) {
        const float a = s_m[g] == -FLT_MAX ? 0.0f : expf(s_m[g] - MM);
        SS += s_S[g] * a;
        OO += s_O[g][d] * a;
    }
    out[(size_t) h * D + d] = OO / SS;
}

// One warp per row, 8 rows per block, 8 BF16 values (16 bytes) read per lane: 2048 columns = 8 passes. The COLS input columns
// share the row read; each sum follows the order of the COLS = 1 case (compiled separately so the single-token path keeps its code).
template <int COLS>
__global__ void k_gemv_bf16(const __nv_bfloat16 * w, const float * x, float * y, int ncols, int nrows) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= nrows) return;
    const uint4 * wr = (const uint4 *) (w + (size_t) row * ncols);
    float s[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) s[c] = 0.0f;
    for (int i = lane; i < ncols / 8; i += 32) {
        const uint4 pk = wr[i];
        const __nv_bfloat162 * b = (const __nv_bfloat162 *) &pk;
        const float2 w0 = __bfloat1622float2(b[0]), w1 = __bfloat1622float2(b[1]);
        const float2 w2 = __bfloat1622float2(b[2]), w3 = __bfloat1622float2(b[3]);
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            const float4 * xv = (const float4 *) (x + (size_t) c * ncols);
            const float4 x0 = xv[2 * i], x1 = xv[2 * i + 1];
            s[c] += w0.x * x0.x + w0.y * x0.y + w1.x * x0.z + w1.y * x0.w + w2.x * x1.x + w2.y * x1.y + w3.x * x1.z +
                    w3.y * x1.w;
        }
    }
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
        float v = s[c];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) y[(size_t) c * nrows + row] = v;
    }
}

// One warp: each lane holds 8 of the 256 experts in registers; sigmoid, then k argmax rounds via shuffles (ties: smaller index,
// as ggml's argsort), weight = p / max(Σ p, 6.1035e-5). n_expert = 256, k <= 8.
__global__ void k_route_topk(const float * logits, int32_t * ids, float * w, int k) {
    constexpr int PER = 8;
    const int lane = threadIdx.x;
    float p[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) p[i] = 1.0f / (1.0f + expf(-logits[i * 32 + lane]));
    float sel_v = 0.0f, sum = 0.0f;
    int sel_i = 0;
    for (int r = 0; r < k; ++r) {
        float v = -FLT_MAX;
        int idx = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            if (p[i] > v) { v = p[i]; idx = i * 32 + lane; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, v, o);
            const int oi = __shfl_xor_sync(0xffffffff, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if ((idx & 31) == lane) p[idx >> 5] = -FLT_MAX;
        if (lane == r) { sel_v = v; sel_i = idx; }
        sum += v;
    }
    if (lane < k) {
        ids[lane] = sel_i;
        w[lane] = sel_v / fmaxf(sum, 6.103515625e-5f);
    }
}

__global__ void k_swiglu(const float * g, const float * u, float * act, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float x = g[i];
    act[i] = x / (1.0f + expf(-x)) * u[i];
}

// Epilogue of mul_mat_vec_q with GLU fusion (ggml mmvq.cu:837-846, zero biases, ggml_cuda_op_silu_single).
__global__ void k_glu_mmvq(const float * g, const float * u, float * act, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float result = u[i];
    result += 0.0f;
    const float gate_value = g[i] + 0.0f;
    result *= gate_value / (1.0f + expf(-gate_value));
    act[i] = result;
}

// Same F32 operation sequence as build_amax_src1_f16: scale(1/1024), log, scale(1/ln2), ceil, clamp, scale(ln2), exp.
__global__ void k_amax_guard(float * act, float * escale, int n) {
    __shared__ float sh[32];
    float * r = act + (size_t) blockIdx.x * n;
    float m = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, fabsf(r[i]));
    m = block_max(m, sh);
    float e = m * (1.0f / 1024.0f);
    e = logf(e);
    e = e * 1.4426950408889634f;
    e = ceilf(e);
    e = fminf(fmaxf(e, 0.0f), 127.0f);
    e = e * 0.6931471805599453f;
    e = expf(e);
    for (int i = threadIdx.x; i < n; i += blockDim.x) r[i] = r[i] / e;
    if (threadIdx.x == 0) escale[blockIdx.x] = e;
}

// amax_guard, then ggml's quantize_q8_1 on act [rows][512]: one block of 512 threads per row (one expert). Writes e and the
// quantized row to xq; 512 columns = 16 Q8_1 blocks, the row MMVQ expects (multiple of 512, no padding).
__global__ void __launch_bounds__(512) k_amax_guard_q8_1(const float * act, float * escale, void * xq, int n) {
    __shared__ float sh[32];
    const float * r = act + (size_t) blockIdx.x * n;
    const float xr = r[threadIdx.x];
    float m = block_max(fabsf(xr), sh);
    float e = m * (1.0f / 1024.0f);
    e = logf(e);
    e = e * 1.4426950408889634f;
    e = ceilf(e);
    e = fminf(fmaxf(e, 0.0f), 127.0f);
    e = e * 0.6931471805599453f;
    e = expf(e);
    const float xi = xr / e;
    if (threadIdx.x == 0) escale[blockIdx.x] = e;
    float amax = fabsf(xi), sum = xi;
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    block_q8_1 * y = (block_q8_1 *) xq + (size_t) blockIdx.x * (n / 32);
    const float d = amax / 127.0f;
    y[threadIdx.x / 32].qs[threadIdx.x % 32] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    if (threadIdx.x % 32 == 0) y[threadIdx.x / 32].ds = make_half2(d, sum);
}

__global__ void k_moe_combine(const float * dexp, const float * escale, const float * w, const float * sh, float * x,
                              int n, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float moe = (dexp[i] * escale[0]) * w[0];
    for (int j = 1; j < k; ++j) moe = moe + (dexp[(size_t) j * n + i] * escale[j]) * w[j];
    x[i] = (moe + sh[i]) + x[i];
}

__global__ void k_add(float * x, const float * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = y[i] + x[i];
}

// --- Token selection: two-pass top-k (argmax = top-1) ---
// Total order: logit descending, then index ascending (ties: smaller index, as std::max_element). Pass 1: each of the kSelBlocks
// blocks keeps the first k allowed elements of its slice of x, in that order. The global top-k lies in the union of the local
// top-ks, so pass 2 (k_sample over the candidates, same order) gives exactly the result of a search over all of x. One block
// for the whole vocabulary would serialize the scan.
constexpr int kSampleMaxK = 64;
constexpr int kSelBlocks = 128;
constexpr int kSelThreads = 256;
__device__ float g_cand_v[kSelBlocks * kSampleMaxK];
__device__ int g_cand_i[kSelBlocks * kSampleMaxK];

// Best (v, i) pair of the block in total order; all threads receive it. sv/si: 32 shared entries.
__device__ void block_best(float & v, int & idx, float * sv, int * si) {
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, v, o);
        const int oi = __shfl_xor_sync(0xffffffff, idx, o);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();  // sv/si may still be read by the previous round
    if (lane == 0) { sv[warp] = v; si[warp] = idx; }
    __syncthreads();
    v = lane < (int) (blockDim.x >> 5) ? sv[lane] : -FLT_MAX;
    idx = lane < (int) (blockDim.x >> 5) ? si[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, v, o);
        const int oi = __shfl_xor_sync(0xffffffff, idx, o);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
}

__global__ void __launch_bounds__(kSelThreads) k_topk_part(const float * x, int n, const uint32_t * mask, int k) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const int chunk = (n + kSelBlocks - 1) / kSelBlocks;
    const int lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
    float prev_v = FLT_MAX;
    int prev_i = -1;
    for (int r = 0; r < k; ++r) {
        float v = -FLT_MAX;
        int idx = 0x7fffffff;
        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            if (mask && !((mask[i >> 5] >> (i & 31)) & 1u)) continue;
            const float xi = x[i];
            const bool after = xi < prev_v || (xi == prev_v && i > prev_i);  // strictly after the previous candidate
            if (after && (xi > v || (xi == v && i < idx))) { v = xi; idx = i; }
        }
        block_best(v, idx, sv, si);
        if (threadIdx.x == 0) {
            g_cand_v[blockIdx.x * k + r] = v;
            g_cand_i[blockIdx.x * k + r] = idx;
        }
        prev_v = v;
        prev_i = idx;
    }
}

// --- Sampling ---
// Philox 4x32-10, adapted from Strata (src/kernels/cuda/sampler.cu:39-64, MIT): u in [0, 1) as a function of (seed, counter).
__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        const uint32_t hi0 = __umulhi(0x9E3779B9u, c0), hi1 = __umulhi(0xBB67AE85u, c2);
        const uint32_t lo0 = 0x9E3779B9u * c0, lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ (uint32_t) i, n2 = hi0 ^ c3;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// Pass 2: one block of 1024 threads over the kSelBlocks * k candidates of k_topk_part (value g_cand_v, original index g_cand_i;
// 0x7fffffff: empty). Final candidates: the first k in total order (k <= 64), by k argmax rounds over the block: round r finds
// the largest element strictly after candidate r - 1 in this order. Then, in llama.cpp order: top-p over the sorted candidates,
// temperature, Philox draw (seed, counter). temperature <= 0: argmax (the first candidate).
__global__ void __launch_bounds__(1024) k_sample(int k, float top_p, float temperature, uint64_t seed,
                                                 uint64_t counter, int32_t * out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float cand_v[kSampleMaxK];
    __shared__ int cand_i[kSampleMaxK];
    const int n = kSelBlocks * k;
    float prev_v = FLT_MAX;
    int prev_i = -1;
    for (int r = 0; r < k; ++r) {
        float v = -FLT_MAX;
        int idx = 0x7fffffff;
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            const int i = g_cand_i[j];
            if (i == 0x7fffffff) continue;
            const float xi = g_cand_v[j];
            const bool after = xi < prev_v || (xi == prev_v && i > prev_i);
            if (after && (xi > v || (xi == v && i < idx))) { v = xi; idx = i; }
        }
        block_best(v, idx, sv, si);
        if (threadIdx.x == 0) { cand_v[r] = v; cand_i[r] = idx; }
        prev_v = v;
        prev_i = idx;
        if (prev_i == 0x7fffffff) break;  // no allowed token left
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    int nk = 0;
    while (nk < k && cand_i[nk] != 0x7fffffff) ++nk;  // fewer than k allowed tokens
    if (nk == 0) { *out = -1; return; }
    if (temperature <= 0.0f || nk == 1) { *out = cand_i[0]; return; }
    // Candidate probabilities (temperature-1 softmax for the top-p cut, as in llama.cpp).
    double p[kSampleMaxK], sum = 0.0;
    for (int j = 0; j < nk; ++j) sum += p[j] = exp((double) cand_v[j] - (double) cand_v[0]);
    int keep = nk;
    double cum = 0.0;
    for (int j = 0; j < nk; ++j) {
        cum += p[j] / sum;
        if (cum >= top_p) { keep = j + 1; break; }
    }
    double w[kSampleMaxK], wsum = 0.0;
    for (int j = 0; j < keep; ++j) wsum += w[j] = exp(((double) cand_v[j] - (double) cand_v[0]) / temperature);
    const double u = philox_uniform(seed, counter) * wsum;
    double acc = 0.0;
    int pick = keep - 1;
    for (int j = 0; j < keep; ++j) {
        acc += w[j];
        if (u < acc) { pick = j; break; }
    }
    *out = cand_i[pick];
}

// Presence penalty, as in rec (kernels/masked_sampling.py): x[i] -= penalty * seen[i].
__global__ void k_presence_apply(float * x, const float * seen, int n, float penalty) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] -= penalty * seen[i];
}

__global__ void k_presence_mark(float * seen, const int32_t * token) {
    if (*token >= 0) seen[*token] = 1.0f;
}

// Column c of a verification window: the tokens assumed chosen before it (draft[0 .. c)) count as seen. Same operation as
// k_presence_apply (s is 0 or 1): same bits as the sequential path.
__global__ void k_presence_apply_col(float * x, const float * seen, const int32_t * draft, int c, int n, float penalty) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = seen[i];
    for (int k = 0; k < c; ++k)
        if (draft[k] == i) s = 1.0f;
    x[i] -= penalty * s;
}

__global__ void k_presence_mark_n(float * seen, const int32_t * tokens, int n) {
    const int k = threadIdx.x;
    if (k < n && tokens[k] >= 0) seen[tokens[k]] = 1.0f;
}

__global__ void k_io_advance(int32_t * io, int pos) {
    io[0] = io[2] < 0 ? 0 : io[2];  // -1 (empty mask): the forward launched in advance reads a valid token; its output is discarded
    io[1] = pos;
}

int blocks(int n) { return (n + kBlock - 1) / kBlock; }

}  // namespace

void swap_bytes(void * a, void * b, size_t n, cudaStream_t st) {
    if (!n) return;
    const size_t fragments = n / 16 + (n % 16 != 0);
    const size_t needed = fragments / kBlock + (fragments % kBlock != 0);
    const unsigned int grid = (unsigned int) (needed < 65535 ? needed : 65535);
    if ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) % 16 == 0)
        k_swap_bytes<true><<<grid, kBlock, 0, st>>>((uint8_t *) a, (uint8_t *) b, n);
    else
        k_swap_bytes<false><<<grid, kBlock, 0, st>>>((uint8_t *) a, (uint8_t *) b, n);
    FNS_CHECK(cudaGetLastError());
}

void argmax(const float * x, int n, int32_t * out, cudaStream_t st, const uint32_t * mask) {
    k_topk_part<<<kSelBlocks, kSelThreads, 0, st>>>(x, n, mask, 1);
    k_sample<<<1, 1024, 0, st>>>(1, 1.0f, 0.0f, 0, 0, out);
    FNS_CHECK(cudaGetLastError());
}

void sample(const float * x, int n, const uint32_t * mask, int top_k, float top_p, float temperature, uint64_t seed,
            uint64_t counter, int32_t * out, cudaStream_t st) {
    const int k = top_k <= 0 || top_k > kSampleMaxK ? kSampleMaxK : top_k;
    k_topk_part<<<kSelBlocks, kSelThreads, 0, st>>>(x, n, mask, k);
    k_sample<<<1, 1024, 0, st>>>(k, top_p, temperature, seed, counter, out);
    FNS_CHECK(cudaGetLastError());
}

void presence_apply(float * x, const float * seen, int n, float penalty, cudaStream_t st) {
    k_presence_apply<<<blocks(n), kBlock, 0, st>>>(x, seen, n, penalty);
    FNS_CHECK(cudaGetLastError());
}

void presence_mark(float * seen, const int32_t * token, cudaStream_t st) {
    k_presence_mark<<<1, 1, 0, st>>>(seen, token);
    FNS_CHECK(cudaGetLastError());
}

void presence_apply_col(float * x, const float * seen, const int32_t * draft, int c, int n, float penalty,
                        cudaStream_t st) {
    k_presence_apply_col<<<blocks(n), kBlock, 0, st>>>(x, seen, draft, c, n, penalty);
    FNS_CHECK(cudaGetLastError());
}

void presence_mark_n(float * seen, const int32_t * tokens, int n, cudaStream_t st) {
    k_presence_mark_n<<<1, 32, 0, st>>>(seen, tokens, n);
    FNS_CHECK(cudaGetLastError());
}

void io_advance(int32_t * io, int pos, cudaStream_t st) {
    k_io_advance<<<1, 1, 0, st>>>(io, pos);
    FNS_CHECK(cudaGetLastError());
}

void embed_q8_0(const void * table, const int32_t * token, float * out, int n, int cols, cudaStream_t st) {
    k_embed_q8_0<<<dim3(blocks(n), cols), kBlock, 0, st>>>((const uint8_t *) table, token, out, n);
    FNS_CHECK(cudaGetLastError());
}

void rms_norm_mul(const float * x, const float * w, float * out, int n, int rows, float eps, cudaStream_t st) {
    k_rms_norm_mul<<<rows, n >= 1024 ? 1024 : 128, 0, st>>>(x, w, out, n, eps);
    FNS_CHECK(cudaGetLastError());
}

void rms_norm_q8_1(const float * x, const float * w, float * out, void * xq, int n, float eps, int rows,
                   cudaStream_t st) {
    if (n != 2048) {
        std::fprintf(stderr, "rms_norm_q8_1: expected n = 2048\n");
        std::abort();
    }
    k_rms_norm_q8_1<<<rows, 1024, 0, st>>>(x, w, out, xq, n, eps);
    FNS_CHECK(cudaGetLastError());
}

void rope_neox(float * x, int n_heads, int head_dim, const int32_t * pos, float base, cudaStream_t st) {
    const float theta_scale = powf(base, -2.0f / head_dim);
    k_rope_neox<<<n_heads, head_dim / 2, 0, st>>>(x, head_dim, pos, theta_scale);
    FNS_CHECK(cudaGetLastError());
}

void store_kv(const float * k, const float * v, void * kc, void * vc, const int32_t * pos, int slots, int n_kv,
              cudaStream_t st) {
    k_store_kv<<<blocks(n_kv), kBlock, 0, st>>>(k, v, (__half *) kc, (__half *) vc, pos, slots, n_kv);
    FNS_CHECK(cudaGetLastError());
}

void qk_norm_rope_store(float * q, float * k, const float * v, const float * qn, const float * kn, void * kc, void * vc,
                        const int32_t * pos, int n_head, int n_head_kv, int head_dim, int slots, float eps,
                        float rope_base, bool rope, int cols, int stride, cudaStream_t st, bool q8) {
    if (head_dim != 128) {
        std::fprintf(stderr, "qk_norm_rope_store: expected head_dim 128\n");
        std::abort();
    }
    const float theta_scale = powf(rope_base, -2.0f / head_dim);
    k_qk_norm_rope_store<<<dim3(n_head + n_head_kv, cols), 128, 0, st>>>(
        q, k, v, qn, kn, (__half *) kc, (__half *) vc, pos, n_head, n_head_kv, slots, eps, theta_scale, rope ? 1 : 0,
        stride, q8);
    FNS_CHECK(cudaGetLastError());
}

void attention(const float * q, const void * kc, const void * vc, const float * sinks, float * out, float * part,
               int n_head, int n_head_kv, int head_dim, const int32_t * pos, int window, int slots, int max_ctx,
               float scale, int cols, int q_stride, cudaStream_t st, bool q8, bool mma) {
    if (head_dim != 128 || n_head != kG * n_head_kv) {
        std::fprintf(stderr, "attention: expected head_dim 128 and %d query heads per KV head\n", kG);
        std::abort();
    }
    // The grid covers the maximum context (fixed in the CUDA graph); chunks past pos exit at once.
    const int span = window > 0 ? window : max_ctx;
    const int n_chunk = (span + kChunk - 1) / kChunk;
    // Partials of one column use the single-token layout; columns are spaced by the full partial size.
    const int part_stride = (int) attention_part_floats(n_head, head_dim, max_ctx > span ? max_ctx : span);
    float * part_o = part;
    float2 * part_ms = (float2 *) (part + (size_t) n_head * n_chunk * head_dim);
    const dim3 grid(n_chunk * cols, n_head_kv);
    if (mma && !q8 && window == 0) {
        attention_mma_part(q, kc, vc, part_o, part_ms, pos, scale, q_stride, part_stride, cols, n_chunk, st);
    } else if (q8) {
        k_attention_part<true><<<grid, kAttWarps * 32, 0, st>>>(q, kc, vc, part_o, part_ms, n_head_kv, pos, window,
                                                                slots, scale, q_stride, part_stride, cols, n_chunk);
    } else {
        k_attention_part<false><<<grid, kAttWarps * 32, 0, st>>>(q, kc, vc, part_o, part_ms, n_head_kv, pos, window,
                                                                 slots, scale, q_stride, part_stride, cols, n_chunk);
    }
    FNS_CHECK(cudaGetLastError());
    k_attention_reduce<<<dim3(n_head, cols), 4 * head_dim, 0, st>>>(part_o, part_ms, sinks, out, n_chunk, pos, window,
                                                                    part_stride);
    FNS_CHECK(cudaGetLastError());
}

size_t attention_part_floats(int n_head, int head_dim, int max_ctx) {
    const int n_chunk = (max_ctx + kChunk - 1) / kChunk;
    return (size_t) n_head * n_chunk * (head_dim + 2);
}

void gemv_bf16(const void * w, const float * x, float * y, int ncols, int nrows, int cols, cudaStream_t st) {
    if (cols < 1 || cols > 8) {
        std::fprintf(stderr, "gemv_bf16: expected 1 to 8 columns\n");
        std::abort();
    }
    const dim3 g((nrows + 7) / 8);
    const __nv_bfloat16 * wb = (const __nv_bfloat16 *) w;
    switch (cols) {
        case 1: k_gemv_bf16<1><<<g, 256, 0, st>>>(wb, x, y, ncols, nrows); break;
        case 2: k_gemv_bf16<2><<<g, 256, 0, st>>>(wb, x, y, ncols, nrows); break;
        case 3: k_gemv_bf16<3><<<g, 256, 0, st>>>(wb, x, y, ncols, nrows); break;
        case 4: k_gemv_bf16<4><<<g, 256, 0, st>>>(wb, x, y, ncols, nrows); break;
        case 5: k_gemv_bf16<5><<<g, 256, 0, st>>>(wb, x, y, ncols, nrows); break;
        case 6: k_gemv_bf16<6><<<g, 256, 0, st>>>(wb, x, y, ncols, nrows); break;
        case 7: k_gemv_bf16<7><<<g, 256, 0, st>>>(wb, x, y, ncols, nrows); break;
        default: k_gemv_bf16<8><<<g, 256, 0, st>>>(wb, x, y, ncols, nrows); break;
    }
    FNS_CHECK(cudaGetLastError());
}

void route_topk(const float * logits, int32_t * ids, float * w, int n_expert, int k, cudaStream_t st) {
    if (n_expert != 256 || k > 8) {
        std::fprintf(stderr, "route_topk: expected 256 experts and k <= 8\n");
        std::abort();
    }
    k_route_topk<<<1, 32, 0, st>>>(logits, ids, w, k);
    FNS_CHECK(cudaGetLastError());
}

void amax_guard_q8_1(const float * act, float * escale, void * xq, int n, int rows, cudaStream_t st) {
    if (n != 512) {
        std::fprintf(stderr, "amax_guard_q8_1: expected n = 512\n");
        std::abort();
    }
    k_amax_guard_q8_1<<<rows, 512, 0, st>>>(act, escale, xq, n);
    FNS_CHECK(cudaGetLastError());
}

void swiglu(const float * g, const float * u, float * act, int n, int rows, cudaStream_t st) {
    k_swiglu<<<blocks(n * rows), kBlock, 0, st>>>(g, u, act, n * rows);
    FNS_CHECK(cudaGetLastError());
}

void glu_mmvq(const float * g, const float * u, float * act, int n, cudaStream_t st) {
    k_glu_mmvq<<<blocks(n), kBlock, 0, st>>>(g, u, act, n);
    FNS_CHECK(cudaGetLastError());
}

void amax_guard(float * act, float * escale, int n, int rows, cudaStream_t st) {
    k_amax_guard<<<rows, 256, 0, st>>>(act, escale, n);
    FNS_CHECK(cudaGetLastError());
}

void moe_combine(const float * dexp, const float * escale, const float * w, const float * sh, float * x, int n, int k,
                 cudaStream_t st) {
    k_moe_combine<<<blocks(n), kBlock, 0, st>>>(dexp, escale, w, sh, x, n, k);
    FNS_CHECK(cudaGetLastError());
}

void add_inplace(float * x, const float * y, int n, cudaStream_t st) {
    k_add<<<blocks(n), kBlock, 0, st>>>(x, y, n);
    FNS_CHECK(cudaGetLastError());
}

__global__ void k_count_routes(const int32_t * ids, uint32_t * counts, int k) {
    if ((int) threadIdx.x < k) atomicAdd(&counts[ids[threadIdx.x]], 1u);
}

void count_routes(const int32_t * ids, uint32_t * counts, int k, cudaStream_t st) {
    k_count_routes<<<1, 32, 0, st>>>(ids, counts, k);
    FNS_CHECK(cudaGetLastError());
}

}  // namespace fns::k
