// src/kernels.cuh - simple kernels of the reference path: readable before fast, one block per row or per head,
// F32 reductions. Arithmetic follows the Flint graph of llama.cpp 19ab5a5e (src/models/flint.cpp,
// src/llama-graph.cpp build_moe_ffn / build_amax_src1_f16); references are cited at each kernel.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

struct cublasContext;  // cublasHandle_t

namespace fns::k {

// Kernels that depend on the token or position read them from GPU memory (`token`, `pos`): a token's CUDA graph is
// captured once and replayed with other values. W-column verification: `cols` consecutive columns, column c reads
// token[c] and position *pos + c; each column's arithmetic is that of cols = 1.

// Row `token[c]` of token_embd (Q8_0, on GPU: blocks of 32 = half d + int8[32]) -> out[c] F32, c < cols.
// ggml dequantize_row_q8_0.
void embed_q8_0(const void * table, const int32_t * token, float * out, int n, int cols, cudaStream_t st);

// out[r][i] = x[r][i] * rsqrt(mean(x[r]^2) + eps) * w[i]. ggml rms_norm then mul (norm.cu:164, eps inside the root).
void rms_norm_mul(const float * x, const float * w, float * out, int n, int rows, float eps, cudaStream_t st);

// rms_norm_mul of `rows` rows of 2048, then ggml's quantize_q8_1 (bit-exact) of each output into xq (rows of
// q8_1_row_bytes(2048)), in one launch.
void rms_norm_q8_1(const float * x, const float * w, float * out, void * xq, int n, float eps, int rows,
                   cudaStream_t st);

// NEOX RoPE (pairs i, i + head_dim/2; Flint is in the NEOX list of llama-model.cpp:3565-3646),
// theta_i = pos * base^(-2 i / head_dim). ggml-cuda rope.cu:123-197, without YaRN or freq_factors.
void rope_neox(float * x, int n_heads, int head_dim, const int32_t * pos, float base, cudaStream_t st);

// K, V F32 -> half cache at slot `*pos % slots`.
void store_kv(const float * k, const float * v, void * kc, void * vc, const int32_t * pos, int slots, int n_kv,
              cudaStream_t st);

// rms_norm_mul per head of Q and K, rope_neox if `rope`, store_kv of K and V: the same operations in one launch.
// Column c: q, k, v offset by c * stride floats (output [cols][Q | K | V] of the QKV GEMV), position *pos + c.
// q8 (opt-in): Q8_0 cache [slots][n_head_kv][4 blocks] instead of half (quantize_f32_q8_0_block of ggml).
void qk_norm_rope_store(float * q, float * k, const float * v, const float * qn, const float * kn, void * kc, void * vc,
                        const int32_t * pos, int n_head, int n_head_kv, int head_dim, int slots, float eps,
                        float rope_base, bool rope, int cols, int stride, cudaStream_t st, bool q8 = false);

// Attention of one token with sinks (ggml soft_max + sinks, ggml-cpu ops.cpp:5980-5993):
//   s_j = scale * q·k_j ;  m = max(max_j s_j, sink_h) ;  out = Σ exp(s_j - m) v_j / (Σ exp(s_j - m) + exp(sink_h - m))
// Positions read: [max(0, pos - window + 1), pos] (window = 0: whole context), slot p % slots.
// GQA: query head h reads KV head h / (n_head / n_head_kv). Split-K over chunks of 256 positions, online softmax;
// `part`: cols * attention_part_floats(n_head, head_dim, max(window, max_ctx)) floats. head_dim = 128 only.
// Column c: q at q + c * q_stride, position *pos + c, output out + c * n_head * head_dim. The KV of the cols positions
// must be written beforehand: each column reads only positions <= its own (causal). q8: Q8_0 cache (component
// d * q, then the same F32 computation).
void attention(const float * q, const void * kc, const void * vc, const float * sinks, float * out, float * part,
               int n_head, int n_head_kv, int head_dim, const int32_t * pos, int window, int slots, int max_ctx,
               float scale, int cols, int q_stride, cudaStream_t st, bool q8 = false, bool mma = false);
size_t attention_part_floats(int n_head, int head_dim, int max_ctx);
// Tensor-core (mma) variant of the global-layer partials, F16 inputs; same layout as attention_part_floats.
void attention_mma_part(const float * q, const void * kc, const void * vc, float * part_o, float2 * part_ms,
                        const int32_t * pos, float scale, int q_stride, int part_stride, int cols, int chunks,
                        cudaStream_t st);

// *out = argmax(x[0..n)); ties: smallest index. One block of 1024 threads. If mask is non-null (uint32 bitmask in
// GPU memory, xgrammar), only indices whose bit is 1 compete.
void argmax(const float * x, int n, int32_t * out, cudaStream_t st, const uint32_t * mask = nullptr);

// Token sampling (as in llama.cpp: top-k, top-p, temperature, draw), among tokens allowed by mask (may be null).
// top_k <= 0 or > 64: 64 candidates. temperature <= 0: argmax. Philox draw (seed, counter).
// *out = -1 if no token is allowed.
void sample(const float * x, int n, const uint32_t * mask, int top_k, float top_p, float temperature, uint64_t seed,
            uint64_t counter, int32_t * out, cudaStream_t st);

// Presence penalty: x[i] -= penalty * seen[i] (seen: 0 or 1 per token, GPU); presence_mark sets
// seen[*token] to 1 (token in GPU memory, ignored if -1).
void presence_apply(float * x, const float * seen, int n, float penalty, cudaStream_t st);
void presence_mark(float * seen, const int32_t * token, cudaStream_t st);
// W-column verification: presence_apply for column c, with draft[0 .. c) (GPU) counted as seen;
// presence_mark_n marks tokens[0 .. n) (GPU, n <= 32).
void presence_apply_col(float * x, const float * seen, const int32_t * draft, int c, int n, float penalty,
                        cudaStream_t st);
void presence_mark_n(float * seen, const int32_t * tokens, int n, cudaStream_t st);

// Chained decode: io[0] = io[2] (token chosen at the previous step, kept on GPU; 0 if it is -1), io[1] = pos.
void io_advance(int32_t * io, int pos, cudaStream_t st);

// y[c][r] = Σ_i W_bf16[r][i] x[c][i] in F32, c < cols (router, ggml's mul_mat_vec_f).
void gemv_bf16(const void * w, const float * x, float * y, int ncols, int nrows, int cols, cudaStream_t st);

// Flint routing: p = sigmoid(logits), top-k on p, weights = p[sel] / max(Σ p[sel], 6.1035e-5) (llama-graph.cpp:2245-2348).
void route_topk(const float * logits, int32_t * ids, float * w, int n_expert, int k, cudaStream_t st);

// counts[ids[j]] += 1, j < k (per-layer routing histogram, expert cache profile).
void count_routes(const int32_t * ids, uint32_t * counts, int k, cudaStream_t st);

// act[r][i] = silu(g[r][i]) * u[r][i], r < rows (ggml swiglu_split).
void swiglu(const float * g, const float * u, float * act, int n, int rows, cudaStream_t st);

// act[i] = (u[i] + 0) * silu(g[i] + 0), i < n: ggml's fused GLU epilogue of the MMVQs (mmvq.cu:837-846, zero biases),
// to get the bits of mmvq_swiglu from two separate GEMVs. act may alias u.
void glu_mmvq(const float * g, const float * u, float * act, int n, cudaStream_t st);

// ggml F16 guard of llama.cpp (llama-graph.cpp:1670-1705) on act [rows][n]: e_r = 2^clamp(ceil(log2(amax_r/1024)),0,127)
// computed by the same sequence of F32 operations; act_r /= e_r; escale[r] = e_r.
void amax_guard(float * act, float * escale, int n, int rows, cudaStream_t st);

// amax_guard (act unmodified) then ggml's quantize_q8_1 of the result into xq [rows][16 blocks]. n = 512.
void amax_guard_q8_1(const float * act, float * escale, void * xq, int n, int rows, cudaStream_t st);

// MoE sum in graph order (llama-graph.cpp:2543-2572, then flint.cpp:170-174):
//   moe = Σ_j (dexp_j * e_j) * w_j  (additions in order j = 0..k-1); x = (moe + sh) + x
void moe_combine(const float * dexp, const float * escale, const float * w, const float * sh, float * x, int n, int k,
                 cudaStream_t st);

// x += y
void add_inplace(float * x, const float * y, int n, cudaStream_t st);

// --- Batched prefill (prefill_kernels.cu): T tokens at positions p0 .. p0 + T - 1, same per-row operations ---
void embed_q8_0_batch(const void * table, const int32_t * ids, float * out, int n, int n_tok, cudaStream_t st);
// Q, K, V of the batch in qkv [T][qkv_stride] (QKV GEMM output); Q -> q_out [T][n_head][128], K/V -> cache.
void qk_norm_rope_store_batch(const float * qkv, int qkv_stride, float * q_out, const float * qn, const float * kn,
                              void * kc, void * vc, int p0, int n_tok, int n_head, int n_head_kv, int head_dim,
                              int slots, float eps, float rope_base, bool rope, cudaStream_t st, bool q8 = false);
// Q8_0 cache [rows][n_head_kv][4 blocks] -> half [rows][n_head_kv][128] (ggml's dequantize_q8_0, d * q in F32, then
// half), rows 0 .. rows - 1: F16 input of the prefill FlashAttention.
void dequant_kv_q8_0(const void * q8, void * f16, int rows, int n_head_kv, cudaStream_t st);
// Opt-in (changes bits): prefill FlashAttention of a global layer, Q·K in INT8 (mma s8) and P·V in F16
// (src/fa_int8.cu). q [n_tok][32][128] F32, half K/V cache [rows][4][128] (rows multiple of 64; rows beyond
// p0 + n_tok - 1 are ignored), sinks [32], out [n_tok][32][128]. Per-layer state: quantized Q (fa_int8_q_bytes),
// quantized and smoothed K (fa_int8_k_bytes), mean of K (4 x 128 floats) set at batch p0 = 0, rows already quantized.
struct FaInt8State {
    void * q = nullptr;      // shared between layers (rewritten on each call)
    void * k = nullptr;      // per layer
    float * mean = nullptr;  // per layer
    int quantized = 0;       // rows of K already quantized
};
size_t fa_int8_q_bytes(int n_tok);
size_t fa_int8_k_bytes(int rows);
size_t fa_int8_stat_floats();  // FaInt8State::mean: mean, per-channel equalization, maxima
void fa_int8_prefill(const float * q, const void * kc, const void * vc, const float * sinks, float * out, int n_tok,
                     int p0, int rows, float scale, FaInt8State & s, cudaStream_t st);
// Opt-in fa_swa (changes bits): prefill FlashAttention of a sliding layer, Q·K and P·V in f16 mma (src/fa_int8.cu),
// only tiles of visible positions (window `window`) are read from the half K/V ring [slots][4][128].
void fa_swa_prefill(const float * q, const void * kc, const void * vc, const float * sinks, float * out, int n_tok,
                    int p0, int slots, int window, float scale, cudaStream_t st);
// Causal mask (and window if window > 0) half [n_tok][n_kv] over cache slots (ring of `slots`).
void prefill_mask(void * mask, int n_kv, int p0, int n_tok, int slots, int window, cudaStream_t st);
// Batch router: y [n_tok][nrows] = W_bf16 · bf16(x[t]) via cuBLAS (F32 accumulation), as llama.cpp does for a batch.
// xb: n_tok * ncols bf16 workspace.
void gemm_bf16_batch(::cublasContext * h, const void * w, const float * x, void * xb, float * y, int ncols,
                     int nrows, int n_tok, cudaStream_t st);
void route_topk_batch(const float * logits, int32_t * ids, float * w, int n_expert, int k, int n_tok, cudaStream_t st);
void moe_combine_batch(const float * dexp, const float * escale, const float * w, const float * sh, float * x, int n,
                       int k, int n_tok, cudaStream_t st);

}  // namespace fns::k
