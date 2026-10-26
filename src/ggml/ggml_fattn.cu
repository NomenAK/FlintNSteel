// src/ggml/ggml_fattn.cu - ggml-cuda MMA F16 FlashAttention for batched prefill, on our own buffers.
//
// Includes ggml/src/ggml-cuda/fattn-mma-f16.cuh from llama.cpp 19ab5a5e65736a35437e6b1c0ccd6562fed01b1b (MIT, The ggml
// authors) and instantiates the single case used here: DKQ = DV = 128, 8 q heads per kv head (ncols2 = 8, the GQA
// optimization chosen by fattn.cu:266-269), ncols1 = 8 (64 columns per tile, the batch > 32 / ncols2 case of fattn.cu:190).
// The kernel is launched by ggml's launch_fattn (fattn-common.cuh:980-1314) with ggml_tensor descriptors over our
// buffers: Q F32 [128, T, 32], K and V F16 [128, n_kv, 4] (layer cache), mask F16 [n_kv, T] (0 / -inf: causal
// and sliding window), sinks F32 [32], output F32 [128, 32, T]. No softcap, no ALiBi.
#include "fattn-mma-f16.cuh"

#include "fns/ggml_kernels.h"

DECL_FATTN_MMA_F16_CASE(128, 128, 8, 8);

// Non-template fattn.cu symbols referenced by fattn-common.cuh. The sparse path does not exist for DKQ = 128
// (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse is false at compile time), so these are never called.
bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int, const ggml_tensor *, const int, const int) {
    return false;
}
void ggml_cuda_flash_attn_ext_compact_mask(const ggml_tensor *, int32_t *, int32_t *, int32_t, int32_t, int32_t,
                                           cudaStream_t) {
    GGML_ABORT("fattn: sparse path not supported");
}

namespace fns {

namespace {

void set_tensor(ggml_tensor & t, ggml_type type, void * data, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1,
                size_t nb2) {
    t = ggml_tensor{};
    t.type = type;
    t.data = data;
    t.ne[0] = ne0;
    t.ne[1] = ne1;
    t.ne[2] = ne2;
    t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = nb1;
    t.nb[2] = nb2;
    t.nb[3] = nb2 * ne2;
}

}  // namespace

void flash_attn_prefill(const float * q, const void * kc, const void * vc, const void * mask, const float * sinks,
                        float * out, int n_tok, int n_kv, int n_head, int n_head_kv, int head_dim, float scale,
                        cudaStream_t stream) {
    GGML_ASSERT(head_dim == 128 && n_head == 8 * n_head_kv);
    GGML_ASSERT(n_kv % FATTN_KQ_STRIDE == 0);
    static ggml_backend_cuda_context ctx(0);
    ctx.streams[0][0] = stream;  // launch_fattn launches on ctx.stream()
    ggml_tensor Q, K, V, M, S, dst;
    set_tensor(Q, GGML_TYPE_F32, (void *) q, head_dim, n_tok, n_head, (size_t) n_head * head_dim * sizeof(float),
               head_dim * sizeof(float));
    const size_t kv_row = (size_t) n_head_kv * head_dim * sizeof(half);
    set_tensor(K, GGML_TYPE_F16, (void *) kc, head_dim, n_kv, n_head_kv, kv_row, head_dim * sizeof(half));
    set_tensor(V, GGML_TYPE_F16, (void *) vc, head_dim, n_kv, n_head_kv, kv_row, head_dim * sizeof(half));
    set_tensor(M, GGML_TYPE_F16, (void *) mask, n_kv, n_tok, 1, (size_t) n_kv * sizeof(half), (size_t) n_kv * n_tok * sizeof(half));
    set_tensor(S, GGML_TYPE_F32, (void *) sinks, n_head, 1, 1, n_head * sizeof(float), n_head * sizeof(float));
    set_tensor(dst, GGML_TYPE_F32, out, head_dim, n_head, n_tok, head_dim * sizeof(float),
               (size_t) n_head * head_dim * sizeof(float));
    dst.op = GGML_OP_FLASH_ATTN_EXT;
    dst.src[0] = &Q;
    dst.src[1] = &K;
    dst.src[2] = &V;
    dst.src[3] = &M;
    dst.src[4] = &S;
    std::memcpy(&dst.op_params[0], &scale, sizeof(float));  // scale; max_bias = softcap = 0
    ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 8, 8>(ctx, &dst);
    ctx.streams[0][0] = nullptr;  // the context does not destroy our stream
}

}  // namespace fns
