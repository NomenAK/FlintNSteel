// src/ggml/ggml_mmq.cu - ggml-cuda quantized GEMMs (MMQ) for batched prefill, on our own buffers.
//
// Dense: ggml's mul_mat_q_case (mmq.cuh; instances in template-instances/mmq-instance-*.cu of llama.cpp
// 19ab5a5e65736a35437e6b1c0ccd6562fed01b1b, MIT, The ggml authors), with the arguments ggml_cuda_mul_mat_q computes
// (mmq.cu:134-274). Experts: mul_mat_q_table (src/ggml/mmq_table.cuh, generated), where expert e's matrix is
// table[e], with the arguments of ggml_cuda_mul_mat_q's MUL_MAT_ID path (mmq.cu:276-353): mm_ids_helper sorts
// (token, expert) pairs by expert, quantize_mmq_q8_1 packs activations in that order, and the GEMM writes each row
// to its original position (ids_dst).
#include "mmq.cuh"
#include "mmid.cuh"
#include "mmq_table.cuh"
#include "quantize.cuh"

#include "fns/ggml_kernels.h"

namespace fns {

namespace {

ggml_backend_cuda_context & cuda_ctx() {
    static ggml_backend_cuda_context ctx(0);
    return ctx;
}

// Picks ggml's tile width J (mmq.cu:189-225) for `ncols_opt` columns.
void best_J(ggml_type type, int64_t nrows_x, int64_t ncols_opt, int & J_best, int & nthreads_best) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    const bool fallback = ggml_cuda_mmq_needs_fallback(nrows_x);
    J_best = 0;
    nthreads_best = 0;
    int ntiles_J_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, GGML_PREC_Q8);
        if (config.type == GGML_TYPE_COUNT) continue;
        if (mmq_get_nbytes_shared(config, cc) > smpbo) continue;
        const int ntiles_x = (int) ((ncols_opt + config.J - 1) / config.J);
        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            nthreads_best = config.nthreads;
            ntiles_J_best = ntiles_x;
        }
    }
    GGML_ASSERT(J_best > 0);
}

template <ggml_type T> void run_case(const char * const * table, uint32_t off, const mmq_args & a, cudaStream_t st) {
    if (table) {
        if (ggml_cuda_mmq_needs_fallback(a.nrows_x)) mul_mat_q_table_switch_J<T, true, GGML_PREC_Q8>(cuda_ctx(), table, off, a, st);
        else mul_mat_q_table_switch_J<T, false, GGML_PREC_Q8>(cuda_ctx(), table, off, a, st);
    } else {
        mul_mat_q_case<T, GGML_PREC_Q8>(cuda_ctx(), a, st);
    }
}

void run(ggml_type type, const char * const * table, uint32_t off, const mmq_args & a, cudaStream_t st) {
    switch (type) {
        case GGML_TYPE_Q8_0: run_case<GGML_TYPE_Q8_0>(table, off, a, st); break;
        case GGML_TYPE_Q4_K: run_case<GGML_TYPE_Q4_K>(table, off, a, st); break;
        case GGML_TYPE_Q5_K: run_case<GGML_TYPE_Q5_K>(table, off, a, st); break;
        case GGML_TYPE_Q6_K: run_case<GGML_TYPE_Q6_K>(table, off, a, st); break;
        case GGML_TYPE_IQ4_XS: run_case<GGML_TYPE_IQ4_XS>(table, off, a, st); break;
        case GGML_TYPE_IQ3_S: run_case<GGML_TYPE_IQ3_S>(table, off, a, st); break;
        case GGML_TYPE_IQ3_XXS: run_case<GGML_TYPE_IQ3_XXS>(table, off, a, st); break;
        case GGML_TYPE_IQ2_S: run_case<GGML_TYPE_IQ2_S>(table, off, a, st); break;
        default: GGML_ABORT("mmq: type %s not supported", ggml_type_name(type));
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

size_t mmq_q8_1_bytes(int64_t ncols, int64_t nrows) {
    // As in ggml: nrows * ncols_padded block_q8_1_mmq, plus one tile of slack (reads go up to J - 1 columns past the end).
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    return (size_t) nrows * padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ + 128 * sizeof(block_q8_1_mmq) + 4096;
}

void mmq(ggml_type type, const void * w, int64_t ncols, int64_t nrows, const float * x, int64_t n_tok, void * xq,
         float * dst, cudaStream_t st) {
    // ggml_cuda_mul_mat_q without ids: src0 [ncols, nrows], src1 [ncols, n_tok] contiguous F32, dst [nrows, n_tok].
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    int J = 0, nth = 0;
    best_J(type, nrows, n_tok, J, nth);
    quantize_mmq_q8_1_cuda(x, nullptr, xq, type, ncols, ncols, ncols * n_tok, ncols * n_tok, padded, n_tok, 1, 1, st);
    CUDA_CHECK(cudaGetLastError());
    const int64_t s01 = ncols / ggml_blck_size(type);
    const int64_t s12 = n_tok * padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const mmq_args a = {(const char *) w, type, (const int *) xq, nullptr, nullptr, dst, nullptr,
                        ncols, nrows, n_tok, s01, n_tok, nrows,
                        1, 1, 0, s12, nrows * n_tok,
                        1, 1, 0, s12, nrows * n_tok,
                        n_tok, J};
    run(type, nullptr, 0, a, st);
}

void mmq_experts(ggml_type type, const char * const * table, size_t off, int64_t ncols, int64_t nrows, int64_t n_expert,
                 const float * x, int64_t x_rows_per_tok, int64_t n_tok, const int32_t * ids, int64_t n_used,
                 MmqIdsScratch & s, float * dst, cudaStream_t st) {
    // ggml_cuda_mul_mat_q with ids: src0 [ncols, nrows, n_expert] (here via table), src1 [ncols, x_rows_per_tok, n_tok]
    // (1: the same row for all n_used experts, gate/up; n_used: one row per expert, down), ids [n_used, n_tok],
    // dst [nrows, n_used, n_tok].
    GGML_ASSERT(x_rows_per_tok == 1 || x_rows_per_tok == n_used);
    const int64_t padded = GGML_PAD(ncols, MATRIX_ROW_PADDING);
    const int64_t n_rows_flat = n_tok * n_used;
    int J = 0, nth = 0;
    best_J(type, nrows, n_tok, J, nth);
    const bool dedup = x_rows_per_tok == 1 && n_used > 1;
    ggml_cuda_launch_mm_ids_helper(ids, s.ids_src1, s.ids_dst, s.expert_bounds, (int) n_expert, (int) n_tok, (int) n_used,
                                   (int) x_rows_per_tok, (int) n_used, (int) x_rows_per_tok, dedup, st);
    CUDA_CHECK(cudaGetLastError());
    if (dedup) {
        quantize_scatter_mmq_q8_1_cuda(x, s.ids_src1, s.xq, type, ncols, /*stride_token=*/ncols, padded, n_tok,
                                       n_rows_flat, (int) n_used, st);
    } else {
        quantize_mmq_q8_1_cuda(x, s.ids_src1, s.xq, type, ncols, ncols, ncols * x_rows_per_tok, ncols * n_rows_flat,
                               padded, n_rows_flat, 1, 1, st);
    }
    CUDA_CHECK(cudaGetLastError());
    const int64_t s01 = ncols / ggml_blck_size(type);
    const int64_t s12 = x_rows_per_tok * padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const mmq_args a = {nullptr, type, (const int *) s.xq, s.ids_dst, s.expert_bounds, dst, nullptr,
                        ncols, nrows, n_rows_flat, s01, n_rows_flat, nrows,
                        n_expert, n_expert, 0, s12, nrows * n_used,
                        1, 1, 0, s12 * n_tok, nrows * n_rows_flat,
                        n_tok, J};
    run(type, table, (uint32_t) off, a, st);
}

}  // namespace fns
