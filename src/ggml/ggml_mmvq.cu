// src/ggml/ggml_mmvq.cu - ggml-cuda quantized GEMV (MMVQ) and Q8_1 activation quantization, called on our own
// buffers without a ggml graph or backend.
//
// Includes ggml/src/ggml-cuda/mmvq.cu from llama.cpp 19ab5a5e65736a35437e6b1c0ccd6562fed01b1b unmodified
// (MIT, Copyright (c) 2023-2026 The ggml authors) to expose its `static` functions (mul_mat_vec_q_switch_type).
// Only the call sites are ours; arguments match those computed by ggml_cuda_mul_mat_vec_q (mmvq.cu:1455-1587).
#include "mmvq.cu"

#include "fns/ggml_kernels.h"

namespace fns {

size_t q8_1_row_bytes(int64_t ncols) {
    return (size_t) (GGML_PAD(ncols, MATRIX_ROW_PADDING) / QK8_1) * sizeof(block_q8_1);
}

void quantize_q8_1(const float * x, void * xq, int64_t ncols, int64_t nrows, cudaStream_t stream) {
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    quantize_row_q8_1_cuda(x, nullptr, xq, GGML_TYPE_F32, ncols, ncols, ncols * nrows, ncols * nrows,
                           padded, nrows, 1, 1, stream);
}

void mmvq(ggml_type type, const void * w, int64_t ncols, int64_t nrows, const void * xq, float * dst, int64_t n_cols,
          cudaStream_t stream) {
    // n_cols input columns: xq [n_cols][s11 blocks], dst [n_cols][nrows]. Chunks of at most 4 columns:
    // 1 to 4 columns keep the GENERIC table at 4 warps (calc_nwarps, mmvq.cu:452-467), so the K-loop split and
    // summation order match a single token. ggml's 5 to 8 column launches use 2 warps and reorder the sums, so chunks stop at 4.
    const int64_t blocks_per_row = ncols / ggml_blck_size(type);
    const int64_t s11 = GGML_PAD(ncols, MATRIX_ROW_PADDING) / QK8_1;
    ggml_cuda_mm_fusion_args_device fusion{};
    for (int64_t c0 = 0; c0 < n_cols; c0 += 4) {
        const int64_t n = std::min<int64_t>(4, n_cols - c0);
        mul_mat_vec_q_switch_type(w, type, (const block_q8_1 *) xq + c0 * s11, nullptr, fusion, dst + c0 * nrows,
            (int) ncols, (int) nrows, (int) n, (int) blocks_per_row, (int) s11, (int) nrows,
            1, 1, 1, (int) (blocks_per_row * nrows), (int) s11, (int) nrows,
            1, 1, 0, 0, 0, 0, stream);
    }
}

void mmvq_ids(ggml_type type, const void * w, int64_t ncols, int64_t nrows, int64_t n_mats,
              const void * xq, int64_t n_xq, const int32_t * ids, int64_t n_ids, float * dst,
              cudaStream_t stream) {
    // MUL_MAT_ID, one token: src0 [ncols, nrows, n_mats], src1 [ncols, n_xq, 1], ids [n_ids, 1],
    // dst [nrows, n_ids, 1]. Output channel c reads matrix ids[c] and activation row c % n_xq.
    const int64_t blocks_per_row = ncols / ggml_blck_size(type);
    const int64_t s11 = GGML_PAD(ncols, MATRIX_ROW_PADDING) / QK8_1;
    ggml_cuda_mm_fusion_args_device fusion{};
    mul_mat_vec_q_switch_type(w, type, xq, ids, fusion, dst, (int) ncols,
        (int) nrows, 1, (int) blocks_per_row, (int) (n_xq * s11), (int) (nrows * n_ids),
        (int) n_mats, (int) n_xq, (int) n_ids, (int) (blocks_per_row * nrows), (int) s11, (int) nrows,
        1, 1, 0, 0, 0, (int) n_ids, stream);
}

void mmvq_swiglu(ggml_type type, const void * w_gate, const void * w_up, int64_t ncols, int64_t nrows,
                 const void * xq, float * dst, cudaStream_t stream) {
    // ggml GLU fusion (ggml-cuda.cu:4317-4354): dst = (up * x) * silu(gate * x), one pass over x.
    const int64_t blocks_per_row = ncols / ggml_blck_size(type);
    const int64_t s11 = GGML_PAD(ncols, MATRIX_ROW_PADDING) / QK8_1;
    ggml_cuda_mm_fusion_args_device fusion{};
    fusion.gate = w_gate;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;
    mul_mat_vec_q_switch_type(w_up, type, xq, nullptr, fusion, dst, (int) ncols,
        (int) nrows, 1, (int) blocks_per_row, (int) s11, (int) nrows,
        1, 1, 1, (int) (blocks_per_row * nrows), (int) s11, (int) nrows,
        1, 1, 0, 0, 0, 0, stream);
}

void mmvq_ids_swiglu_shared(ggml_type type, const void * w_gate, const void * w_up, int64_t ncols, int64_t nrows,
                            int64_t n_mats, const void * xq, const int32_t * ids, int64_t n_ids, float * dst,
                            const void * sh_gate, const void * sh_up, float * sh_dst, cudaStream_t stream) {
    // ggml "shared expert" fusion (ggml-cuda.cu:3786-3801): one launch computes the n_ids routed experts
    // (dst [n_ids][nrows]) and the shared expert (sh_dst [nrows]), which occupies the last grid channel. Both
    // must have the same type and shape (the caller checks, as ggml_cuda_match_shared_expert does).
    const int64_t blocks_per_row = ncols / ggml_blck_size(type);
    const int64_t s11 = GGML_PAD(ncols, MATRIX_ROW_PADDING) / QK8_1;
    ggml_cuda_mm_fusion_args_device fusion{};
    fusion.gate = w_gate;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;
    if (sh_dst) {
        fusion.shared_up = sh_up;
        fusion.shared_gate = sh_gate;
        fusion.shared_dst = sh_dst;
        fusion.shared_stride_col_dst = (uint32_t) nrows;
    }
    mul_mat_vec_q_switch_type(w_up, type, xq, ids, fusion, dst, (int) ncols,
        (int) nrows, 1, (int) blocks_per_row, (int) s11, (int) (nrows * n_ids),
        (int) n_mats, 1, (int) n_ids, (int) (blocks_per_row * nrows), (int) s11, (int) nrows,
        1, 1, 0, 0, 0, (int) n_ids, stream);
}

// --- Experts via a pointer table -------------------------------------------------------------------------------
// Transcription of mul_mat_vec_q (mmvq.cu:621-874, ncols_dst = 1, GENERIC table: 4 warps) where expert c's matrix
// comes from `table[ids[c]] + off` instead of `vx + ids[c] * stride_channel_x`: a slot in VRAM, or a pinned host
// copy read by the GPU (zero-copy). Same K-loop split and summation order (per-thread partials, then warps
// 1..nwarps-1 added to warp 0, then warp_reduce_sum) and same GLU. Results are bit-identical to mmvq_ids and
// mmvq_ids_swiglu_shared (checked by tests/expert_parity.cu).
// nwarps < 4 (deviation from ggml): when a row needs only qi/vdr threads per qk block and everything fits in one
// pass, the warps with no ggml work would only add exact zeros, so they are not launched (gate+up IQ3_XXS on
// 2048 columns: 64 useful threads of 128; down IQ4_XS on 512 columns: 16 of 128).

// ggml's IQ3_XXS / IQ3_S products (vecdotq.cuh:1235-1311: same arithmetic, same operations in the same order), with
// the grid read from `grid` (a shared-memory copy of iq3xxs_grid / iq3s_grid with identical values) instead of
// global memory: grid-dependent loads no longer go through L1/L2.
static __device__ __forceinline__ float vec_dot_iq3_xxs_sgrid(const void * __restrict__ vbq,
                                                              const block_q8_1 * __restrict__ bq8_1, const int & kbx,
                                                              const int & iqs, const uint32_t * grid) {
    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(grid[q3[l0 + 0]], grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq3_s_sgrid(const void * __restrict__ vbq,
                                                            const block_q8_1 * __restrict__ bq8_1, const int & kbx,
                                                            const int & iqs, const uint32_t * grid) {
    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

template <ggml_type type, bool glu, int rows_per_block, int nwarps, bool sgrid = false>
__launch_bounds__(nwarps * WARP_SIZE, 1)
static __global__ void k_mmvq_experts(const char * const * table, uint32_t off_x, uint32_t off_g,
                                      const block_q8_1 * y0, uint32_t stride_y, uint32_t n_xq, uint32_t x_div,
                                      const int32_t * ids, float * dst, uint32_t ncols, uint32_t nrows) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_block * blockIdx.x;
    const int blocks_per_row_x = ncols / qk;
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
    const uint32_t c = blockIdx.y;
    const char * base = table[ids[c]];
    const void * vx = base + off_x;
    [[maybe_unused]] const void * vg = base + off_g;
    const block_q8_1 * y = y0 + ((c / x_div) % n_xq) * stride_y;
    const int stride_row_x = blocks_per_row_x;
    const int kbx_offset = row0 * stride_row_x;
    float tmp[rows_per_block] = {0.0f};
    float tmp_gate[rows_per_block] = {0.0f};
    if constexpr (sgrid) {
        constexpr int n_grid = type == GGML_TYPE_IQ3_XXS ? 256 : 512;
        __shared__ uint32_t s_grid[n_grid];
        for (int i = tid; i < n_grid; i += nwarps * WARP_SIZE)
            s_grid[i] = type == GGML_TYPE_IQ3_XXS ? iq3xxs_grid[i] : iq3s_grid[i];
        __syncthreads();
        for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk / QK8_1);
            const int kqs = vdr * (tid % (qi / vdr));
#pragma unroll
            for (int i = 0; i < rows_per_block; ++i) {
                const int kb = kbx_offset + i * stride_row_x + kbx;
                if constexpr (type == GGML_TYPE_IQ3_XXS) {
                    tmp[i] += vec_dot_iq3_xxs_sgrid(vx, &y[kby], kb, kqs, s_grid);
                    if constexpr (glu) tmp_gate[i] += vec_dot_iq3_xxs_sgrid(vg, &y[kby], kb, kqs, s_grid);
                } else {
                    tmp[i] += vec_dot_iq3_s_sgrid(vx, &y[kby], kb, kqs, s_grid);
                    if constexpr (glu) tmp_gate[i] += vec_dot_iq3_s_sgrid(vg, &y[kby], kb, kqs, s_grid);
                }
            }
        }
    } else {
        for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk / QK8_1);
            const int kqs = vdr * (tid % (qi / vdr));
#pragma unroll
            for (int i = 0; i < rows_per_block; ++i) {
                tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs);
                if constexpr (glu) tmp_gate[i] += vec_dot_q_cuda(vg, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs);
            }
        }
    }

    if constexpr (nwarps > 1) {
        __shared__ float tmp_shared[nwarps - 1][rows_per_block][WARP_SIZE];
        [[maybe_unused]] __shared__ float tmp_shared_gate[glu ? nwarps - 1 : 1][rows_per_block][WARP_SIZE];
        if (threadIdx.y > 0) {
#pragma unroll
            for (int i = 0; i < rows_per_block; ++i) {
                tmp_shared[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
                if constexpr (glu) tmp_shared_gate[threadIdx.y - 1][i][threadIdx.x] = tmp_gate[i];
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) return;
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l) {
                tmp[i] += tmp_shared[l][i][threadIdx.x];
                if constexpr (glu) tmp_gate[i] += tmp_shared_gate[l][i][threadIdx.x];
            }
        }
    }

    dst += (size_t) c * nrows + row0;
#pragma unroll
    for (int i = 0; i < rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<WARP_SIZE>(tmp[i]);
        if constexpr (glu) tmp_gate[i] = warp_reduce_sum<WARP_SIZE>(tmp_gate[i]);
        if (threadIdx.x == i && (rows_per_block == 1 || uint32_t(row0 + i) < nrows)) {
            float result = tmp[i];
            if constexpr (glu) {
                // ggml adds the fusion biases (zero here) before the GLU: same operations, same signed zeros.
                result += 0.0f;
                result *= ggml_cuda_op_silu_single(tmp_gate[i] + 0.0f);
            }
            dst[i] = result;
        }
    }
}

// Short rows (a single K-loop pass, `lanes` < 32 useful threads per row, e.g. expert down on 512 columns: 16).
// Each group of `lanes` threads is one ggml one-warp block, reduced over `lanes` threads instead of 32. In ggml,
// the other 32 - lanes threads of the warp have no terms: their partial sums stay 0, and the first reduction steps
// (offsets >= lanes) add 0.0f. The (x + 0.0f) additions are kept, then the reduction runs over `lanes`: same
// operations, same bits. A warp holds 32 / lanes groups; a block holds `nwarps` warps.
template <ggml_type type, int rows_per_group, int lanes, int nwarps>
__launch_bounds__(nwarps * WARP_SIZE, 1)
static __global__ void k_mmvq_experts_short(const char * const * table, uint32_t off_x, const block_q8_1 * y0,
                                            uint32_t stride_y, uint32_t n_xq, uint32_t x_div, const int32_t * ids,
                                            float * dst, uint32_t ncols, uint32_t nrows) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
    constexpr int groups_per_warp = WARP_SIZE / lanes;

    const int lane = threadIdx.x % lanes;
    const int group = (threadIdx.y * groups_per_warp + threadIdx.x / lanes);
    const int row0 = rows_per_group * (blockIdx.x * nwarps * groups_per_warp + group);
    const int blocks_per_row_x = ncols / qk;
    const uint32_t c = blockIdx.y;
    const void * vx = table[ids[c]] + off_x;
    const block_q8_1 * y = y0 + ((c / x_div) % n_xq) * stride_y;
    const int kbx_offset = row0 * blocks_per_row_x;

    float tmp[rows_per_group] = {0.0f};
    // Single pass (guaranteed by the caller): thread `lane` handles qk block lane / (qi / vdr).
    const int kbx = lane / (qi / vdr);
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (lane % (qi / vdr));
#pragma unroll
    for (int i = 0; i < rows_per_group; ++i) {
        tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i * blocks_per_row_x + kbx, kqs);
    }

    dst += (size_t) c * nrows + row0;
#pragma unroll
    for (int i = 0; i < rows_per_group; ++i) {
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset >= lanes; offset >>= 1) tmp[i] += 0.0f;
        tmp[i] = warp_reduce_sum<lanes>(tmp[i]);
        if (lane == i && uint32_t(row0 + i) < nrows) dst[i] = tmp[i];
    }
}

template <ggml_type type, int rows_per_group, int lanes, int nwarps>
static void launch_short(const char * const * table, size_t off_x, const void * xq, int64_t n_xq, int64_t x_div,
                         const int32_t * ids, int64_t n_ids, float * dst, int64_t ncols, int64_t nrows,
                         uint32_t stride_y, cudaStream_t stream) {
    constexpr int rows_per_block = rows_per_group * nwarps * (WARP_SIZE / lanes);
    const dim3 grid((uint32_t) ((nrows + rows_per_block - 1) / rows_per_block), (uint32_t) n_ids);
    k_mmvq_experts_short<type, rows_per_group, lanes, nwarps><<<grid, dim3(WARP_SIZE, nwarps), 0, stream>>>(
        table, (uint32_t) off_x, (const block_q8_1 *) xq, stride_y, (uint32_t) n_xq, (uint32_t) x_div, ids, dst,
        (uint32_t) ncols, (uint32_t) nrows);
}

template <ggml_type type, bool glu, int rows_per_block, int nwarps>
static void launch_rows(const char * const * table, size_t off_x, size_t off_g, const void * xq, int64_t n_xq,
                        int64_t x_div, const int32_t * ids, int64_t n_ids, float * dst, int64_t ncols, int64_t nrows,
                        uint32_t stride_y, cudaStream_t stream) {
    const dim3 grid((uint32_t) ((nrows + rows_per_block - 1) / rows_per_block), (uint32_t) n_ids);
    // IQ3_XXS / IQ3_S: grid in shared memory (sgrid).
    constexpr bool sgrid = type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
    k_mmvq_experts<type, glu, rows_per_block, nwarps, sgrid><<<grid, dim3(WARP_SIZE, nwarps), 0, stream>>>(
        table, (uint32_t) off_x, (uint32_t) off_g, (const block_q8_1 *) xq, stride_y, (uint32_t) n_xq,
        (uint32_t) x_div, ids, dst, (uint32_t) ncols, (uint32_t) nrows);
}

template <ggml_type type, bool glu, int rows_per_block>
static void launch_warps(const char * const * table, size_t off_x, size_t off_g, const void * xq, int64_t n_xq,
                         int64_t x_div, const int32_t * ids, int64_t n_ids, float * dst, int64_t ncols, int64_t nrows,
                         uint32_t stride_y, cudaStream_t stream) {
    // Useful threads: (qi / vdr) per qk block of the row. While they fit in fewer than 4 warps, a single K-loop pass
    // suffices with fewer warps, and each thread computes the same terms as in ggml.
    constexpr int qk = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi;
    const int64_t useful = ncols / qk * (qi / get_vdr_mmvq(type));
#define FNS_EXP_ARGS table, off_x, off_g, xq, n_xq, x_div, ids, n_ids, dst, ncols, nrows, stride_y, stream
    if (useful <= WARP_SIZE) launch_rows<type, glu, rows_per_block, 1>(FNS_EXP_ARGS);
    else if (useful <= 2 * WARP_SIZE) launch_rows<type, glu, rows_per_block, 2>(FNS_EXP_ARGS);
    else launch_rows<type, glu, rows_per_block, 4>(FNS_EXP_ARGS);
#undef FNS_EXP_ARGS
}

template <ggml_type type, bool glu>
static void launch_mmvq_experts(const char * const * table, size_t off_x, size_t off_g, const void * xq, int64_t n_xq,
                                int64_t x_div, const int32_t * ids, int64_t n_ids, float * dst, int64_t ncols,
                                int64_t nrows, cudaStream_t stream) {
    // Choice of ggml (mmvq.cu:1142-1177, Ampere, ncols_dst = 1): "small_k" (4 rows per block) when one pass of the
    // block covers the whole row, except IQ3_XXS and IQ3_S. It only changes rows per block, not the sums.
    // Deviation from ggml: down IQ3_S (512 columns, one pass) also uses 4 rows per block. With 1 row, its one-warp
    // block (16 useful threads) caps occupancy at 16 blocks per SM. Gate/up IQ3_* (2048 columns) stay at 1 row;
    // 2 and 4 rows gave no gain.
    constexpr int qk = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi;
    constexpr int per_warp = get_vdr_mmvq(type) * WARP_SIZE / qi;
    const bool one_pass = ncols / qk < 4 * per_warp;
    const bool small_k = one_pass && ((type != GGML_TYPE_IQ3_XXS && type != GGML_TYPE_IQ3_S)
                                      || (type == GGML_TYPE_IQ3_S && !glu));
    const uint32_t stride_y = (uint32_t) (GGML_PAD(ncols, MATRIX_ROW_PADDING) / QK8_1);
    // Expert down on 512 columns (IQ4_XS, IQ3_S, Q4_K: 16 useful threads per row): k_mmvq_experts_short, two ggml rows
    // per warp, 4 warps per block, same bits.
    const int64_t useful = ncols / qk * (qi / get_vdr_mmvq(type));
    if constexpr (!glu) {
        if (useful == 16) {
            if (small_k) launch_short<type, 4, 16, 4>(table, off_x, xq, n_xq, x_div, ids, n_ids, dst, ncols, nrows, stride_y, stream);
            else launch_short<type, 1, 16, 4>(table, off_x, xq, n_xq, x_div, ids, n_ids, dst, ncols, nrows, stride_y, stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
#define FNS_EXP_ARGS table, off_x, off_g, xq, n_xq, x_div, ids, n_ids, dst, ncols, nrows, stride_y, stream
    if (small_k) launch_warps<type, glu, 4>(FNS_EXP_ARGS);
    else launch_warps<type, glu, 1>(FNS_EXP_ARGS);
#undef FNS_EXP_ARGS
    CUDA_CHECK(cudaGetLastError());
}

template <bool glu>
static void switch_mmvq_experts(ggml_type type, const char * const * table, size_t off_x, size_t off_g, const void * xq,
                                int64_t n_xq, int64_t x_div, const int32_t * ids, int64_t n_ids, float * dst,
                                int64_t ncols, int64_t nrows, cudaStream_t stream) {
#define FNS_EXP_CASE(T) \
    case T: launch_mmvq_experts<T, glu>(table, off_x, off_g, xq, n_xq, x_div, ids, n_ids, dst, ncols, nrows, stream); break;
    switch (type) {
        FNS_EXP_CASE(GGML_TYPE_Q8_0)
        FNS_EXP_CASE(GGML_TYPE_Q4_K)
        FNS_EXP_CASE(GGML_TYPE_Q5_K)
        FNS_EXP_CASE(GGML_TYPE_Q6_K)
        FNS_EXP_CASE(GGML_TYPE_IQ4_XS)
        FNS_EXP_CASE(GGML_TYPE_IQ3_S)
        FNS_EXP_CASE(GGML_TYPE_IQ3_XXS)
        FNS_EXP_CASE(GGML_TYPE_IQ2_S)
        default: GGML_ABORT("mmvq_experts: type %s not supported", ggml_type_name(type));
    }
#undef FNS_EXP_CASE
}

void mmvq_experts_swiglu(ggml_type type, const char * const * table, size_t gate_off, size_t up_off, int64_t ncols,
                         int64_t nrows, const void * xq, int64_t n_xq, const int32_t * ids, int64_t n_ids, float * dst,
                         cudaStream_t stream) {
    // n_ids = n_xq * (experts per column): channel c reads input row c / (n_ids / n_xq).
    switch_mmvq_experts<true>(type, table, up_off, gate_off, xq, n_xq, n_ids / n_xq, ids, n_ids, dst, ncols, nrows,
                              stream);
}

void mmvq_experts(ggml_type type, const char * const * table, size_t off, int64_t ncols, int64_t nrows, const void * xq,
                  int64_t n_xq, const int32_t * ids, int64_t n_ids, float * dst, cudaStream_t stream) {
    switch_mmvq_experts<false>(type, table, off, 0, xq, n_xq, 1, ids, n_ids, dst, ncols, nrows, stream);
}

bool mmvq_supports(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ2_S:
            return true;
        default:
            return false;
    }
}

}  // namespace fns
