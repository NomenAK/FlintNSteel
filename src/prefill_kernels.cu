// src/prefill_kernels.cu - batched prefill kernels: the same operations as the token-by-token path (kernels.cu),
// applied to T consecutive tokens at positions p0 .. p0 + T - 1. One block (or warp) per token and per row; each row's
// arithmetic is that of the single-token kernel, cited at each function.
#include "kernels.cuh"

#include <cublas_v2.h>

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

__device__ float block_sum128(float v, float * sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = sh[0] + sh[1] + sh[2] + sh[3];
    __syncthreads();
    return v;
}

// block_q8_0 from ggml-common.h (MIT, The ggml authors); same quantization as store_q8_0 in kernels.cu
// (quantize_f32_q8_0_block of ggml-cuda, cpy-utils.cuh:137-154, llama.cpp 19ab5a5e).
struct block_q8_0 {
    __half d;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 de ggml : 34 octets");

__device__ __forceinline__ void store_q8_0(block_q8_0 * b, float x, int lane) {
    float amax = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    b->qs[lane] = (int8_t) roundf(x * id);
    if (lane == 0) b->d = __float2half(d);
}

// k_embed_q8_0 for T tokens: row ids[t] of token_embd (Q8_0) -> x[t].
__global__ void k_embed_batch(const uint8_t * table, const int32_t * ids, float * out, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t * blk = table + (size_t) ids[t] * (n / 32) * 34 + (i / 32) * 34;
    const float d = __half2float(*(const __half *) blk);
    out[(size_t) t * n + i] = d * (float) (int8_t) blk[2 + (i % 32)];
}

// k_qk_norm_rope_store for T tokens: block (h, t) handles head h of token t at position p0 + t. Q and K are in the
// output [T][Q | K | V] of the QKV GEMM (rows of qkv_stride floats); normalized and rotated Q goes to q_out
// [T][n_head][128] (FlashAttention input); K and V as half into the cache at slot (p0 + t) % slots.
__global__ void __launch_bounds__(128) k_qk_norm_rope_store_batch(const float * qkv, int qkv_stride, float * q_out,
                                                                  const float * qn, const float * kn, __half * kc,
                                                                  __half * vc, int p0, int n_head, int n_head_kv,
                                                                  int slots, float eps, float theta_scale, int rope,
                                                                  bool q8) {
    constexpr int D = 128;
    __shared__ float sh[4];
    __shared__ float row[D];
    const int t = blockIdx.y;
    const bool is_q = blockIdx.x < (unsigned) n_head;
    const int head = is_q ? blockIdx.x : blockIdx.x - n_head;
    const float * src = qkv + (size_t) t * qkv_stride + (is_q ? 0 : n_head * D) + (size_t) head * D;
    const float * w = is_q ? qn : kn;
    const int i = threadIdx.x;
    const float xi = src[i];
    const float s = block_sum128(xi * xi, sh);
    const float scale = rsqrtf(s / D + eps);
    row[i] = scale * xi * w[i];
    __syncthreads();
    const int pos = p0 + t;
    float y = row[i];
    if (rope) {
        const int j = i % (D / 2);
        const float theta = (float) pos * powf(theta_scale, (2 * j) / 2.0f);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = row[j], x1 = row[j + D / 2];
        y = i < D / 2 ? x0 * c - x1 * sn : x0 * sn + x1 * c;
    }
    if (is_q) {
        q_out[((size_t) t * n_head + head) * D + i] = y;
    } else {
        const size_t row = (size_t) (pos % slots) * n_head_kv + head;
        const float vi = qkv[(size_t) t * qkv_stride + (n_head + n_head_kv) * D + (size_t) head * D + i];
        if (q8) {
            store_q8_0((block_q8_0 *) kc + row * (D / 32) + i / 32, y, i & 31);
            store_q8_0((block_q8_0 *) vc + row * (D / 32) + i / 32, vi, i & 31);
        } else {
            kc[row * D + i] = __float2half(y);
            vc[row * D + i] = __float2half(vi);
        }
    }
}

// Q8_0 -> half, one component per thread: ggml's dequantize_q8_0 (d * q in F32), then half.
__global__ void k_dequant_kv_q8_0(const block_q8_0 * q, __half * f, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const block_q8_0 & b = q[i / 32];
    f[i] = __float2half(__half2float(b.d) * (float) b.qs[i % 32]);
}

// Mask [T][n_kv] for queries p0 .. p0 + T - 1 over cache slots 0 .. n_kv - 1: slot s holds the most recent position
// p <= p0 + T - 1 with p % slots == s (ring; slots >= n_kv for the global cache, where p < 0 marks a never-written slot).
// Visible (0) if 0 <= p <= pos and pos - p < window (window = 0: no window), otherwise -inf. llama.cpp window
// (llama-hparams.h:512-517): masked if pos - p >= n_swa. A position overwritten by a more recent position of the batch
// is outside the window of every query of the batch as soon as slots >= window + T - 1: the caller guarantees this.
__global__ void k_prefill_mask(__half * mask, int n_kv, int p0, int slots, int window) {
    const int t = blockIdx.y;
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_kv) return;
    const int pos = p0 + t;
    const int last = p0 + (int) gridDim.y - 1;
    const int p = last - ((last - s) % slots + slots) % slots;
    const bool vis = s < slots && p >= 0 && p <= pos && (window <= 0 || pos - p < window);
    mask[(size_t) t * n_kv + s] = vis ? __float2half(0.0f) : __float2half(-INFINITY);
}

// x F32 -> BF16 (round to nearest), as the src1 conversion of ggml_cuda_mul_mat_batched_cublas.
__global__ void k_f32_to_bf16(const float * x, __nv_bfloat16 * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

// k_route_topk for T tokens: one warp per token.
__global__ void k_route_topk_batch(const float * logits, int32_t * ids, float * w, int k) {
    constexpr int PER = 8;
    const int lane = threadIdx.x;
    const int t = blockIdx.x;
    const float * lg = logits + (size_t) t * 256;
    float p[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) p[i] = 1.0f / (1.0f + expf(-lg[i * 32 + lane]));
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
        ids[(size_t) t * k + lane] = sel_i;
        w[(size_t) t * k + lane] = sel_v / fmaxf(sum, 6.103515625e-5f);
    }
}

// k_moe_combine for T tokens: x[t] = (Σ_j (dexp[t][j] * e[t][j]) * w[t][j] + sh[t]) + x[t].
__global__ void k_moe_combine_batch(const float * dexp, const float * escale, const float * w, const float * sh,
                                    float * x, int n, int k) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float * d = dexp + (size_t) t * k * n;
    const float * e = escale + (size_t) t * k;
    const float * ww = w + (size_t) t * k;
    float moe = (d[i] * e[0]) * ww[0];
    for (int j = 1; j < k; ++j) moe = moe + (d[(size_t) j * n + i] * e[j]) * ww[j];
    x[(size_t) t * n + i] = (moe + sh[(size_t) t * n + i]) + x[(size_t) t * n + i];
}

}  // namespace

void embed_q8_0_batch(const void * table, const int32_t * ids, float * out, int n, int n_tok, cudaStream_t st) {
    k_embed_batch<<<dim3((n + 255) / 256, n_tok), 256, 0, st>>>((const uint8_t *) table, ids, out, n);
    FNS_CHECK(cudaGetLastError());
}

void qk_norm_rope_store_batch(const float * qkv, int qkv_stride, float * q_out, const float * qn, const float * kn,
                              void * kc, void * vc, int p0, int n_tok, int n_head, int n_head_kv, int head_dim,
                              int slots, float eps, float rope_base, bool rope, cudaStream_t st, bool q8) {
    if (head_dim != 128) {
        std::fprintf(stderr, "qk_norm_rope_store_batch: head_dim 128 expected\n");
        std::abort();
    }
    const float theta_scale = powf(rope_base, -2.0f / head_dim);
    k_qk_norm_rope_store_batch<<<dim3(n_head + n_head_kv, n_tok), 128, 0, st>>>(
        qkv, qkv_stride, q_out, qn, kn, (__half *) kc, (__half *) vc, p0, n_head, n_head_kv, slots, eps, theta_scale,
        rope ? 1 : 0, q8);
    FNS_CHECK(cudaGetLastError());
}

void dequant_kv_q8_0(const void * q8, void * f16, int rows, int n_head_kv, cudaStream_t st) {
    const size_t n = (size_t) rows * n_head_kv * 128;
    k_dequant_kv_q8_0<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>((const block_q8_0 *) q8, (__half *) f16, n);
    FNS_CHECK(cudaGetLastError());
}

void prefill_mask(void * mask, int n_kv, int p0, int n_tok, int slots, int window, cudaStream_t st) {
    k_prefill_mask<<<dim3((n_kv + 255) / 256, n_tok), 256, 0, st>>>((__half *) mask, n_kv, p0, slots, window);
    FNS_CHECK(cudaGetLastError());
}

// Batch router: cuBLAS GEMM BF16 x BF16 with F32 accumulation and F32 output, llama.cpp's batch path
// (ggml_cuda_mul_mat_batched_cublas, ggml-cuda.cu:1572-1628: prefer_f32_output for BF16 outside RDNA3/CDNA).
// xb: n_tok * ncols bf16 workspace.
void gemm_bf16_batch(cublasHandle_t h, const void * w, const float * x, void * xb, float * y, int ncols, int nrows,
                     int n_tok, cudaStream_t st) {
    const int n = ncols * n_tok;
    k_f32_to_bf16<<<(n + 255) / 256, 256, 0, st>>>(x, (__nv_bfloat16 *) xb, n);
    FNS_CHECK(cudaGetLastError());
    const float alpha = 1.0f, beta = 0.0f;
    if (cublasSetStream(h, st) != CUBLAS_STATUS_SUCCESS ||
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, nrows, n_tok, ncols, &alpha, w, CUDA_R_16BF, ncols, xb, CUDA_R_16BF,
                     ncols, &beta, y, CUDA_R_32F, nrows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) !=
            CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemm_bf16_batch: cublasGemmEx failed\n");
        std::abort();
    }
}

void route_topk_batch(const float * logits, int32_t * ids, float * w, int n_expert, int k, int n_tok, cudaStream_t st) {
    if (n_expert != 256 || k > 8) {
        std::fprintf(stderr, "route_topk_batch: 256 experts and k <= 8 expected\n");
        std::abort();
    }
    k_route_topk_batch<<<n_tok, 32, 0, st>>>(logits, ids, w, k);
    FNS_CHECK(cudaGetLastError());
}

void moe_combine_batch(const float * dexp, const float * escale, const float * w, const float * sh, float * x, int n,
                       int k, int n_tok, cudaStream_t st) {
    k_moe_combine_batch<<<dim3((n + 255) / 256, n_tok), 256, 0, st>>>(dexp, escale, w, sh, x, n, k);
    FNS_CHECK(cudaGetLastError());
}

}  // namespace fns::k
