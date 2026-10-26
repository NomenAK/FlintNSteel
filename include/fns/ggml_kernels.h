// include/fns/ggml_kernels.h - ggml-cuda kernels called directly (src/ggml/).
#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace fns {

// Bytes of one row of `ncols` activations quantized to Q8_1, with ggml's padding (512 columns).
size_t q8_1_row_bytes(int64_t ncols);

// x [nrows][ncols] f32 -> xq [nrows][q8_1_row_bytes(ncols)] (ggml's quantize_row_q8_1_cuda).
void quantize_q8_1(const float * x, void * xq, int64_t ncols, int64_t nrows, cudaStream_t stream);

// dst[c][nrows] = W[nrows][ncols] · x[c], c < n_cols, with W in ggml format `type` and x already quantized
// (xq [n_cols][q8_1_row_bytes(ncols)]). ggml launches of at most 4 columns: each column has the bits of a single token.
void mmvq(ggml_type type, const void * w, int64_t ncols, int64_t nrows, const void * xq, float * dst, int64_t n_cols,
          cudaStream_t stream);

// MUL_MAT_ID for one token: dst[c][nrows] = W[ids[c]] · xq[c % n_xq], c < n_ids. W [n_mats][nrows][ncols].
void mmvq_ids(ggml_type type, const void * w, int64_t ncols, int64_t nrows, int64_t n_mats,
              const void * xq, int64_t n_xq, const int32_t * ids, int64_t n_ids, float * dst,
              cudaStream_t stream);

// dst[r] = (W_up · x)[r] * silu((W_gate · x)[r]): ggml's GLU fusion, x read once.
void mmvq_swiglu(ggml_type type, const void * w_gate, const void * w_up, int64_t ncols, int64_t nrows,
                 const void * xq, float * dst, cudaStream_t stream);

// Routed experts (one shared input row) with SwiGLU, and optionally the shared expert in the same launch:
// dst[c] = up[ids[c]]·x * silu(gate[ids[c]]·x), sh_dst = up_sh·x * silu(gate_sh·x). sh_* have the same type and shape.
void mmvq_ids_swiglu_shared(ggml_type type, const void * w_gate, const void * w_up, int64_t ncols, int64_t nrows,
                            int64_t n_mats, const void * xq, const int32_t * ids, int64_t n_ids, float * dst,
                            const void * sh_gate, const void * sh_up, float * sh_dst, cudaStream_t stream);

// Expert pointer tables (expert cache): the matrix [nrows][ncols] of expert e starts at table[e] + off (bytes), in VRAM
// or in pinned, mapped host memory (zero-copy). Results are bit-identical to mmvq_ids_swiglu_shared (without shared
// expert) and to mmvq_ids. `table` is in GPU memory, read by the kernel.
//   swiglu: dst[c] = up_e·x[c / (n_ids / n_xq)] * silu(gate_e·x[…]), e = ids[c]: n_xq input rows, each shared by
//           n_ids / n_xq consecutive experts (n_xq = 1: one token; W columns: n_xq = W).
//   simple: dst[c] = W_e · xq[c % n_xq].
void mmvq_experts_swiglu(ggml_type type, const char * const * table, size_t gate_off, size_t up_off, int64_t ncols,
                         int64_t nrows, const void * xq, int64_t n_xq, const int32_t * ids, int64_t n_ids, float * dst,
                         cudaStream_t stream);
void mmvq_experts(ggml_type type, const char * const * table, size_t off, int64_t ncols, int64_t nrows, const void * xq,
                  int64_t n_xq, const int32_t * ids, int64_t n_ids, float * dst, cudaStream_t stream);

// --- Batched prefill: ggml MMQ (src/ggml/ggml_mmq.cu) ---
// Bytes of the MMQ Q8_1 quantization of nrows rows of ncols activations, padding included.
size_t mmq_q8_1_bytes(int64_t ncols, int64_t nrows);

// dst[t][r] = (W · x[t])[r], t < n_tok; x F32 [n_tok][ncols], xq: mmq_q8_1_bytes(ncols, n_tok) bytes.
void mmq(ggml_type type, const void * w, int64_t ncols, int64_t nrows, const float * x, int64_t n_tok, void * xq,
         float * dst, cudaStream_t stream);

// Scratch buffers of the expert path (ggml MUL_MAT_ID) for at most n_tok tokens and n_used experts per token:
// ids_src1 and ids_dst n_tok*n_used (+128) int32, expert_bounds n_expert+1, xq mmq_q8_1_bytes(ncols_max, n_tok*n_used) bytes.
struct MmqIdsScratch {
    int32_t * ids_src1 = nullptr, * ids_dst = nullptr, * expert_bounds = nullptr;
    void * xq = nullptr;
};

// dst[t][j][r] = (W_e · x[t][j % x_rows_per_tok])[r], e = ids[t][j], j < n_used; the matrix of expert e starts
// at table[e] + off (table in GPU memory). x_rows_per_tok = 1 (gate/up: one row per token) or n_used (down).
void mmq_experts(ggml_type type, const char * const * table, size_t off, int64_t ncols, int64_t nrows, int64_t n_expert,
                 const float * x, int64_t x_rows_per_tok, int64_t n_tok, const int32_t * ids, int64_t n_used,
                 MmqIdsScratch & s, float * dst, cudaStream_t stream);

// ggml's MMA F16 FlashAttention for a batch of n_tok queries (src/ggml/ggml_fattn.cu):
// out[t][h] = softmax(scale * q[t][h] · K^T + mask[t] ; sink_h) · V, K/V half [n_kv][n_head_kv][128] (layer cache,
// n_kv multiple of 256), mask half [n_tok][n_kv] (0 or -inf), q F32 [n_tok][n_head][128], out F32 likewise.
// n_head = 8 * n_head_kv, head_dim = 128.
void flash_attn_prefill(const float * q, const void * kc, const void * vc, const void * mask, const float * sinks,
                        float * out, int n_tok, int n_kv, int n_head, int n_head_kv, int head_dim, float scale,
                        cudaStream_t stream);

// Quantized types FlintNSteel accepts (the MMVQ kernels).
bool mmvq_supports(ggml_type type);

}  // namespace fns
