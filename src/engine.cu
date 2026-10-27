// src/engine.cu - reference path: one token per call, all weights in VRAM in GGUF form, F32 activations, F16 KV.
// The graph follows llama.cpp 19ab5a5e src/models/flint.cpp line by line:
//
//   h   = x + Wo · Attn(RoPE?(RMSNorm_q(Wq·n)), RoPE?(RMSNorm_k(Wk·n)), Wv·n)     n = RMSNorm(x) · attn_norm
//   out = h + FFN(RMSNorm(h) · ffn_norm)                                          dense FFN (layers 0-1) or MoE + shared
//
// Quantized GEMVs are ggml-cuda's (src/ggml/ggml_mmvq.cu); the rest is in kernels.cu.
#include "fns/engine.h"

#include "fns/ggml_kernels.h"
#include "kernels.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define FNS_CHECK(x)                                                                                  \
    do {                                                                                              \
        cudaError_t e_ = (x);                                                                         \
        if (e_ != cudaSuccess) {                                                                      \
            throw std::runtime_error(std::string("CUDA ") + #x + " : " + cudaGetErrorString(e_));    \
        }                                                                                             \
    } while (0)

namespace fns {

namespace {
constexpr size_t kXqBytes = 1 << 20;  // Q8_1 activations: largest input (6144) or 6 rows of 512
}

DeviceMat Engine::upload(const Tensor & t) {
    DeviceMat d;
    d.type = t.type;
    for (int i = 0; i < 4; ++i) d.ne[i] = t.ne[i];
    // ggml reads quantized rows up to a multiple of 512 columns (MATRIX_ROW_PADDING): pad the end of the buffer.
    const size_t pad = 4096;
    FNS_CHECK(cudaMalloc(&d.d, t.nbytes + pad));
    FNS_CHECK(cudaMemset((char *) d.d + t.nbytes, 0, pad));
    FNS_CHECK(cudaMemcpy(d.d, t.data, t.nbytes, cudaMemcpyHostToDevice));
    allocs_.push_back(d.d);
    weight_bytes_ += t.nbytes;
    return d;
}

// Q, K and V concatenated row-wise into one [Q + 2 KV][E] matrix: a single MMVQ with identical per-row results
// (rows of a ggml type are independent; in this model the three share a type in every layer).
DeviceMat Engine::upload_qkv(const Tensor & q, const Tensor & k, const Tensor & v) {
    if (q.type != k.type || q.type != v.type) throw std::runtime_error("Q, K, V have different types: " + q.name);
    DeviceMat d;
    d.type = q.type;
    d.ne[0] = q.ne[0];
    d.ne[1] = q.ne[1] + k.ne[1] + v.ne[1];
    const size_t n = q.nbytes + k.nbytes + v.nbytes, pad = 4096;
    FNS_CHECK(cudaMalloc(&d.d, n + pad));
    FNS_CHECK(cudaMemset((char *) d.d + n, 0, pad));
    FNS_CHECK(cudaMemcpy(d.d, q.data, q.nbytes, cudaMemcpyHostToDevice));
    FNS_CHECK(cudaMemcpy((char *) d.d + q.nbytes, k.data, k.nbytes, cudaMemcpyHostToDevice));
    FNS_CHECK(cudaMemcpy((char *) d.d + q.nbytes + k.nbytes, v.data, v.nbytes, cudaMemcpyHostToDevice));
    allocs_.push_back(d.d);
    weight_bytes_ += n;
    return d;
}

Engine::Engine(const Model & model, const EngineOptions & opt) : m_(model), opt_(opt) {
    const Hparams & hp = m_.hp;
    if (opt_.ring < hp.n_swa) throw std::runtime_error("ring must cover the sliding window (>= 513)");
    if (opt_.verify_cols < 0 || opt_.verify_cols > 8) throw std::runtime_error("verify_cols: 0 to 8");
    if (opt_.attention_mma && opt_.kv_global_q8) throw std::runtime_error("attention_mma requires F16 KV");
    if (opt_.ubatch > 0) {
        // A batch of T positions writes T ring slots: overwritten positions must lie outside the window of every
        // query in the batch, so ring >= n_swa + T - 1. FlashAttention reads tiles of 256 slots.
        opt_.ring = std::max(opt_.ring, hp.n_swa + opt_.ubatch - 1);
        opt_.ring = (opt_.ring + 255) / 256 * 256;
    }
    opt_.ring = std::max(opt_.ring, hp.n_swa + opt_.verify_cols - 1);  // same rule for W columns
    FNS_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    if (const char * p = std::getenv("FNS_TRACE")) trace_layer_ = std::atoi(p);

    output_norm_ = upload(m_.output_norm);
    output_ = upload(m_.output);
    layers_.resize(m_.layers.size());
    const size_t kv_row = (size_t) hp.n_head_kv * hp.head_dim * sizeof(__half);
    for (size_t il = 0; il < m_.layers.size(); ++il) {
        const Layer & s = m_.layers[il];
        DeviceLayer & L = layers_[il];
        L.attn_norm = upload(s.attn_norm);
        L.wqkv = upload_qkv(s.wq, s.wk, s.wv);
        L.wo = upload(s.wo);
        L.q_norm = upload(s.q_norm);
        L.k_norm = upload(s.k_norm);
        L.sinks = upload(s.sinks);
        L.ffn_norm = upload(s.ffn_norm);
        if (s.dense) {
            L.ffn_gate = upload(s.ffn_gate);
            L.ffn_up = upload(s.ffn_up);
            L.ffn_down = upload(s.ffn_down);
        } else {
            L.gate_inp = upload(s.gate_inp);
            L.gate_shexp = upload(s.gate_shexp);
            L.up_shexp = upload(s.up_shexp);
            L.down_shexp = upload(s.down_shexp);
            if (s.gate_exps.type != s.up_exps.type) throw std::runtime_error("gate and up have different types: " + s.gate_exps.name);
            L.gu_type = s.gate_exps.type;
            L.down_type = s.down_exps.type;
            L.gate_bytes = s.gate_exps.nbytes / hp.n_expert;
            L.up_bytes = s.up_exps.nbytes / hp.n_expert;
        }
        L.slots = s.swa ? opt_.ring : opt_.n_ctx;
        L.kv_q8 = !s.swa && opt_.kv_global_q8;
        // Allocated rows: multiple of 256 (prefill FlashAttention tiles), zeroed (masked slots are read; a NaN there
        // would give NaN * 0). Q8_0: 4 blocks of 34 bytes per head (136 instead of 256).
        const size_t rows = (size_t) (L.slots + 255) / 256 * 256;
        const size_t row_bytes = L.kv_q8 ? (size_t) hp.n_head_kv * (hp.head_dim / 32) * 34 : kv_row;
        FNS_CHECK(cudaMalloc(&L.k_cache, row_bytes * rows));
        FNS_CHECK(cudaMalloc(&L.v_cache, row_bytes * rows));
        FNS_CHECK(cudaMemset(L.k_cache, 0, row_bytes * rows));
        FNS_CHECK(cudaMemset(L.v_cache, 0, row_bytes * rows));
        allocs_.push_back(L.k_cache);
        allocs_.push_back(L.v_cache);
    }

    auto falloc = [&](float *& p, size_t n) {
        FNS_CHECK(cudaMalloc(&p, n * sizeof(float)));
        allocs_.push_back(p);
    };
    const int E = hp.n_embd, QD = hp.n_head * hp.head_dim, KVD = hp.n_head_kv * hp.head_dim, K = hp.n_expert_used;
    const size_t C = (size_t) std::max(1, opt_.verify_cols);  // buffer column count
    falloc(x_, C * E);
    falloc(h_, C * E);
    falloc(q_, C * (QD + 2 * KVD));  // GEMV QKV output: [cols][Q | K | V]
    k_ = q_ + QD;
    v_ = k_ + KVD;
    falloc(att_, C * QD);
    falloc(o_, C * E);
    falloc(router_, C * hp.n_expert);
    falloc(wsel_, C * K);
    falloc(escale_, C * K);
    falloc(gate_, C * ((size_t) K * hp.n_ff_exp + hp.n_ff));
    falloc(up_, C * ((size_t) K * hp.n_ff_exp + hp.n_ff));
    falloc(act_, C * ((size_t) K * hp.n_ff_exp + hp.n_ff));
    falloc(dexp_, C * K * E);
    falloc(sh_, C * E);
    falloc(ffn_, C * E);
    falloc(scores_, C * k::attention_part_floats(hp.n_head, hp.head_dim, std::max(opt_.n_ctx, hp.n_swa)));
    falloc(logits_, C * hp.n_vocab);
    FNS_CHECK(cudaMalloc(&ids_, C * K * sizeof(int32_t)));
    allocs_.push_back(ids_);
    FNS_CHECK(cudaMalloc(&xq_, kXqBytes));
    allocs_.push_back(xq_);
    tok_embd_ = upload(m_.tok_embd);
    FNS_CHECK(cudaMalloc(&io_, 4 * sizeof(int32_t)));  // [token, pos, argmax, -]
    allocs_.push_back(io_);
    FNS_CHECK(cudaMallocHost(&io_host_, 4 * sizeof(int32_t)));
    tok_p_ = io_;
    pos_p_ = io_ + 1;
    if (opt_.verify_cols > 0) {
        FNS_CHECK(cudaMalloc(&vio_, 17 * sizeof(int32_t)));  // [p0, ids[8], argmax[8]]
        allocs_.push_back(vio_);
        FNS_CHECK(cudaMallocHost(&vio_host_, 17 * sizeof(int32_t)));
    }
    if (opt_.rebalance_every > 0) opt_.count_routing = true;
    if (opt_.count_routing) {
        const size_t n = m_.layers.size() * hp.n_expert * sizeof(uint32_t);
        FNS_CHECK(cudaMalloc(&route_counts_, n));
        FNS_CHECK(cudaMemset(route_counts_, 0, n));
        allocs_.push_back(route_counts_);
    }
    // Prefill buffers before the experts: they take VRAM from the expert reserve.
    if (opt_.ubatch > 0) alloc_prefill();
    // Experts last: they take the VRAM that remains.
    place_experts();
    if (const char * p = std::getenv("FNS_REBALANCE_TIMING")) {
        if (opt_.rebalance_every > 0 && std::strcmp(p, "1") == 0) {
            placement_.rebalance_timing = true;
            FNS_CHECK(cudaEventCreate(&rebalance_start_));
            FNS_CHECK(cudaEventCreate(&rebalance_end_));
        }
    }
    // The graph is captured only outside diagnostics: trace() reads and prints mid-token.
    use_graph_ = trace_layer_ < 0 && std::getenv("FNS_NO_GRAPH") == nullptr;
}

void Engine::alloc_prefill() {
    const Hparams & hp = m_.hp;
    const size_t T = (size_t) opt_.ubatch, E = hp.n_embd, K = hp.n_expert_used;
    const size_t QD = (size_t) hp.n_head * hp.head_dim, KVD = (size_t) hp.n_head_kv * hp.head_dim;
    const size_t FF = std::max<size_t>((size_t) hp.n_ff, std::max<size_t>(K * hp.n_ff_exp, hp.n_ff_shexp));
    auto fa = [&](float *& p, size_t n) {
        FNS_CHECK(cudaMalloc(&p, n * sizeof(float)));
        allocs_.push_back(p);
    };
    auto ia = [&](int32_t *& p, size_t n) {
        FNS_CHECK(cudaMalloc(&p, n * sizeof(int32_t)));
        FNS_CHECK(cudaMemset(p, 0, n * sizeof(int32_t)));
        allocs_.push_back(p);
    };
    auto va = [&](void *& p, size_t n) {
        FNS_CHECK(cudaMalloc(&p, n));
        FNS_CHECK(cudaMemset(p, 0, n));
        allocs_.push_back(p);
    };
    fa(pf_.x, T * E);
    fa(pf_.h, T * E);
    fa(pf_.qkv, T * (QD + 2 * KVD));
    fa(pf_.q, T * QD);
    fa(pf_.att, T * QD);
    fa(pf_.o, T * E);
    fa(pf_.router, T * hp.n_expert);
    fa(pf_.wsel, T * K);
    fa(pf_.escale, T * K);
    fa(pf_.gate, T * FF);
    fa(pf_.up, T * FF);
    fa(pf_.act, T * FF);
    fa(pf_.dexp, T * K * E);
    fa(pf_.sh, T * E);
    fa(pf_.logits, hp.n_vocab);
    ia(pf_.ids, T * K);
    ia(pf_.tok, T);
    const size_t ncols_max = std::max<size_t>(QD, std::max<size_t>(FF, E));
    va(pf_.xq, fns::mmq_q8_1_bytes((int64_t) ncols_max, (int64_t) T));
    ia(pf_.mm.ids_src1, T * K + 128);
    ia(pf_.mm.ids_dst, T * K + 128);
    ia(pf_.mm.expert_bounds, hp.n_expert + 1);
    va(pf_.mm.xq, fns::mmq_q8_1_bytes((int64_t) std::max<size_t>(E, hp.n_ff_exp), (int64_t) (T * K)));
    pf_.mask_kv = std::max((opt_.n_ctx + 255) / 256 * 256, opt_.ring);
    va(pf_.mask, (size_t) pf_.mask_kv * T * sizeof(__half));
    if (opt_.kv_global_q8) {  // one pair only: global layers are processed one after another
        const size_t bytes = (size_t) pf_.mask_kv * hp.n_head_kv * hp.head_dim * sizeof(__half);
        va(pf_.kf16, bytes);
        va(pf_.vf16, bytes);
    }
    if (opt_.fa_int8) {  // Q shared; quantized K and per-layer mean for each global layer, mask_kv rows
        if (opt_.kv_global_q8) throw std::runtime_error("fa_int8 and kv_global_q8 are incompatible");
        va(fa_q_, k::fa_int8_q_bytes((int) T));
        fa_k_.assign(m_.layers.size(), nullptr);
        fa_mean_.assign(m_.layers.size(), nullptr);
        fa_quantized_.assign(m_.layers.size(), 0);
        for (size_t il = 0; il < m_.layers.size(); ++il) {
            if (m_.layers[il].swa) continue;
            va(fa_k_[il], k::fa_int8_k_bytes(pf_.mask_kv));
            void * m = nullptr;
            va(m, k::fa_int8_stat_floats() * sizeof(float));
            fa_mean_[il] = (float *) m;
        }
    }
    va(pf_.xb, T * E * sizeof(__nv_bfloat16));
    if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
    cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH);
}

int32_t Engine::prefill(const int32_t * ids, int n, int p0, float * logits) {
    if (n <= 0) throw std::runtime_error("prefill: empty batch");
    if (p0 < 0 || p0 + n > opt_.n_ctx) throw std::runtime_error("prefill: positions outside the context");
    if (opt_.ubatch <= 0) {  // no batch buffers: token by token
        int32_t next = 0;
        for (int i = 0; i < n; ++i) {
            if (logits) step(ids[i], p0 + i, logits + (size_t) i * m_.hp.n_vocab);
            else next = step_greedy(ids[i], p0 + i);
        }
        if (logits) {
            const float * l = logits + (size_t) (n - 1) * m_.hp.n_vocab;
            next = (int32_t) (std::max_element(l, l + m_.hp.n_vocab) - l);
        }
        return next;
    }
    for (int i = 0; i < n; i += opt_.ubatch) {
        const int c = std::min(opt_.ubatch, n - i);
        prefill_chunk(ids + i, c, p0 + i, logits ? logits + (size_t) i * m_.hp.n_vocab : nullptr);
    }
    FNS_CHECK(cudaMemcpyAsync(io_host_ + 2, io_ + 2, sizeof(int32_t), cudaMemcpyDeviceToHost, st_));
    FNS_CHECK(cudaStreamSynchronize(st_));
    return io_host_[2];
}

// A batch of T tokens at positions p0 .. p0 + T - 1: the operations of forward() row by row, with ggml's MMQ GEMMs
// and FlashAttention (llama.cpp does the same for ubatch > 1: same kernels, same formulas).
void Engine::prefill_chunk(const int32_t * ids_host, int T, int p0, float * logits) {
    const Hparams & hp = m_.hp;
    const int E = hp.n_embd, D = hp.head_dim, QD = hp.n_head * D, KVD = hp.n_head_kv * D, K = hp.n_expert_used;
    const int F = hp.n_ff_exp, NE = hp.n_expert;
    FNS_CHECK(cudaMemcpyAsync(pf_.tok, ids_host, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, st_));
    k::embed_q8_0_batch(tok_embd_.d, pf_.tok, pf_.x, E, T, st_);
    int mask_for = -1;  // window of the mask already built (0: global, n_swa: sliding)
    for (size_t il = 0; il < layers_.size(); ++il) {
        const DeviceLayer & L = layers_[il];
        const Layer & s = m_.layers[il];
        // Attention.
        k::rms_norm_mul(pf_.x, (const float *) L.attn_norm.d, pf_.h, E, T, hp.rms_eps, st_);
        mmq(L.wqkv.type, L.wqkv.d, E, L.wqkv.ne[1], pf_.h, T, pf_.xq, pf_.qkv, st_);
        k::qk_norm_rope_store_batch(pf_.qkv, QD + 2 * KVD, pf_.q, (const float *) L.q_norm.d, (const float *) L.k_norm.d,
                                    L.k_cache, L.v_cache, p0, T, hp.n_head, hp.n_head_kv, D, L.slots, hp.rms_eps,
                                    hp.rope_base, s.swa, st_, L.kv_q8);
        const int window = s.swa ? hp.n_swa : 0;
        // Columns read: the whole ring (sliding) or positions 0 .. p0 + T - 1 rounded up to 256 (global).
        const int n_kv = s.swa ? L.slots : std::min(pf_.mask_kv, (p0 + T + 255) / 256 * 256);
        // The mask is only used by ggml's FlashAttention (the fa_int8 / fa_swa paths compute their own visibility).
        const bool ggml_fa = s.swa ? !opt_.fa_swa : !opt_.fa_int8;
        if (ggml_fa && (mask_for != window || !s.swa)) {
            k::prefill_mask(pf_.mask, n_kv, p0, T, L.slots, window, st_);
            mask_for = window;
        }
        const void * kc = L.k_cache, * vc = L.v_cache;
        if (L.kv_q8) {  // ggml F16 FlashAttention: global K, V dequantized (as launch_fattn for a Q8_0 cache)
            k::dequant_kv_q8_0(L.k_cache, pf_.kf16, n_kv, hp.n_head_kv, st_);
            k::dequant_kv_q8_0(L.v_cache, pf_.vf16, n_kv, hp.n_head_kv, st_);
            kc = pf_.kf16;
            vc = pf_.vf16;
        }
        if (opt_.fa_int8 && !s.swa) {
            k::FaInt8State fs{fa_q_, fa_k_[il], fa_mean_[il], fa_quantized_[il]};
            k::fa_int8_prefill(pf_.q, L.k_cache, L.v_cache, (const float *) L.sinks.d, pf_.att, T, p0, pf_.mask_kv,
                               1.0f / std::sqrt((float) D), fs, st_);
            fa_quantized_[il] = fs.quantized;
        } else if (opt_.fa_swa && s.swa) {
            k::fa_swa_prefill(pf_.q, L.k_cache, L.v_cache, (const float *) L.sinks.d, pf_.att, T, p0, L.slots,
                              hp.n_swa, 1.0f / std::sqrt((float) D), st_);
        } else {
            flash_attn_prefill(pf_.q, kc, vc, pf_.mask, (const float *) L.sinks.d, pf_.att, T, n_kv,
                               hp.n_head, hp.n_head_kv, D, 1.0f / std::sqrt((float) D), st_);
        }
        mmq(L.wo.type, L.wo.d, QD, E, pf_.att, T, pf_.xq, pf_.o, st_);
        k::add_inplace(pf_.x, pf_.o, T * E, st_);
        // FFN.
        k::rms_norm_mul(pf_.x, (const float *) L.ffn_norm.d, pf_.h, E, T, hp.rms_eps, st_);
        if (s.dense) {
            mmq(L.ffn_gate.type, L.ffn_gate.d, E, hp.n_ff, pf_.h, T, pf_.xq, pf_.gate, st_);
            mmq(L.ffn_up.type, L.ffn_up.d, E, hp.n_ff, pf_.h, T, pf_.xq, pf_.up, st_);
            k::swiglu(pf_.gate, pf_.up, pf_.act, hp.n_ff, T, st_);
            mmq(L.ffn_down.type, L.ffn_down.d, hp.n_ff, E, pf_.act, T, pf_.xq, pf_.o, st_);
            k::add_inplace(pf_.x, pf_.o, T * E, st_);
            continue;
        }
        k::gemm_bf16_batch(cublas_, L.gate_inp.d, pf_.h, pf_.xb, pf_.router, E, NE, T, st_);
        k::route_topk_batch(pf_.router, pf_.ids, pf_.wsel, NE, K, T, st_);
        if (route_counts_) {
            for (int t = 0; t < T; ++t)
                k::count_routes(pf_.ids + (size_t) t * K, route_counts_ + il * NE, K, st_);
        }
        const size_t down_off = L.gate_bytes + L.up_bytes;
        mmq_experts(L.gu_type, L.exp_table, 0, E, F, NE, pf_.h, 1, T, pf_.ids, K, pf_.mm, pf_.gate, st_);
        mmq_experts(L.gu_type, L.exp_table, L.gate_bytes, E, F, NE, pf_.h, 1, T, pf_.ids, K, pf_.mm, pf_.up, st_);
        k::swiglu(pf_.gate, pf_.up, pf_.act, F, T * K, st_);
        k::amax_guard(pf_.act, pf_.escale, F, T * K, st_);
        mmq_experts(L.down_type, L.exp_table, down_off, F, E, NE, pf_.act, K, T, pf_.ids, K, pf_.mm, pf_.dexp, st_);
    // Shared expert: gate, up, SwiGLU, down.
        mmq(L.gate_shexp.type, L.gate_shexp.d, E, hp.n_ff_shexp, pf_.h, T, pf_.xq, pf_.gate, st_);
        mmq(L.up_shexp.type, L.up_shexp.d, E, hp.n_ff_shexp, pf_.h, T, pf_.xq, pf_.up, st_);
        k::swiglu(pf_.gate, pf_.up, pf_.act, hp.n_ff_shexp, T, st_);
        mmq(L.down_shexp.type, L.down_shexp.d, hp.n_ff_shexp, E, pf_.act, T, pf_.xq, pf_.sh, st_);
        k::moe_combine_batch(pf_.dexp, pf_.escale, pf_.wsel, pf_.sh, pf_.x, E, K, T, st_);
    }
    // Head: logits of the last position (or of all, for comparison); argmax of the last in io_[2].
    for (int t = logits ? 0 : T - 1; t < T; ++t) {
        k::rms_norm_mul(pf_.x + (size_t) t * E, (const float *) output_norm_.d, h_, E, 1, hp.rms_eps, st_);
        gemv(output_, h_, logits_, E);
        if (logits) {
            FNS_CHECK(cudaMemcpyAsync(logits + (size_t) t * hp.n_vocab, logits_, (size_t) hp.n_vocab * sizeof(float),
                                      cudaMemcpyDeviceToHost, st_));
        }
    }
    k::argmax(logits_, hp.n_vocab, io_ + 2, st_);
    if (logits) FNS_CHECK(cudaStreamSynchronize(st_));
}

// Placement of routed experts: each expert is a fixed-size slot [gate | up | down] per layer. The most frequent ones per
// the routing profile (default: layer by layer, in order) go to VRAM while the budget allows; the others are copied to
// pinned mapped host memory, which the GPU reads directly (zero-copy). The kernel only sees a pointer table.
void Engine::place_experts() {
    const Hparams & hp = m_.hp;
    const int NE = hp.n_expert, NL = (int) m_.layers.size();
    constexpr size_t kAlign = 256, kPad = 4096;
    std::vector<size_t> slot(NL, 0);
    for (int il = 0; il < NL; ++il) {
        if (m_.layers[il].dense) continue;
        const DeviceLayer & L = layers_[il];
        const size_t db = m_.layers[il].down_exps.nbytes / NE;
        slot[il] = (L.gate_bytes + L.up_bytes + db + kAlign - 1) / kAlign * kAlign;
    }
    size_t free_b = 0, total_b = 0;
    FNS_CHECK(cudaMemGetInfo(&free_b, &total_b));
    // Margin for the table and the per-layer padding, on top of the requested reserve.
    const size_t overhead = (size_t) NL * (kPad + NE * sizeof(char *)) + ((size_t) 64 << 20);
    size_t budget = free_b > opt_.vram_reserve + overhead ? free_b - opt_.vram_reserve - overhead : 0;
    if (opt_.expert_vram >= 0) budget = std::min(budget, (size_t) opt_.expert_vram);

    // Priority order: descending routing count; ties (or no profile): ascending layer, then expert.
    std::vector<uint32_t> prof((size_t) NL * NE, 0);
    if (!opt_.expert_profile.empty()) {
        FILE * f = std::fopen(opt_.expert_profile.c_str(), "rb");
        if (!f || std::fread(prof.data(), sizeof(uint32_t), prof.size(), f) != prof.size())
            throw std::runtime_error("routing profile unreadable or wrong size: " + opt_.expert_profile);
        std::fclose(f);
    }
    std::vector<int> order;
    for (int il = 0; il < NL; ++il)
        if (slot[il])
            for (int e = 0; e < NE; ++e) order.push_back(il * NE + e);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return prof[a] > prof[b]; });
    placement_ = ExpertPlacement{};
    placement_.resident.assign((size_t) NL * NE, 0);
    size_t used = 0;
    for (int i : order) {
        const size_t s = slot[i / NE];
        if (used + s > budget) continue;  // a smaller slot from another layer may still fit
        used += s;
        placement_.resident[i] = 1;
    }

    char ** tables = nullptr;
    FNS_CHECK(cudaMalloc(&tables, (size_t) NL * NE * sizeof(char *)));
    allocs_.push_back(tables);
    exp_tables_ = tables;
    std::vector<char *> host_table((size_t) NL * NE, nullptr);
    slot_host_.assign((size_t) NL * NE, nullptr);
    slot_bytes_ = slot;
    freq_.assign((size_t) NL * NE, 0.0);
    for (size_t i = 0; i < freq_.size(); ++i) freq_[i] = prof[i];
    for (int il = 0; il < NL; ++il) {
        if (!slot[il]) continue;
        const Layer & s = m_.layers[il];
        DeviceLayer & L = layers_[il];
        const size_t gb = L.gate_bytes, ub = L.up_bytes, db = s.down_exps.nbytes / NE;
        int n_dev = 0;
        for (int e = 0; e < NE; ++e) n_dev += placement_.resident[(size_t) il * NE + e];
        const int n_host = NE - n_dev;
        char * dev = nullptr, * host = nullptr;
        if (n_dev) {
            FNS_CHECK(cudaMalloc(&dev, n_dev * slot[il] + kPad));
            FNS_CHECK(cudaMemset(dev + n_dev * slot[il], 0, kPad));
            allocs_.push_back(dev);
        }
        if (n_host) {
            FNS_CHECK(cudaHostAlloc(&host, n_host * slot[il] + kPad, cudaHostAllocMapped | cudaHostAllocWriteCombined));
            std::memset(host + n_host * slot[il], 0, kPad);
            host_allocs_.push_back(host);
        }
        int i_dev = 0, i_host = 0;
        for (int e = 0; e < NE; ++e) {
            const char * g = (const char *) s.gate_exps.data + e * gb;
            const char * u = (const char *) s.up_exps.data + e * ub;
            const char * d = (const char *) s.down_exps.data + e * db;
            char * p;
            if (placement_.resident[(size_t) il * NE + e]) {
                p = dev + (i_dev++) * slot[il];
                FNS_CHECK(cudaMemcpy(p, g, gb, cudaMemcpyHostToDevice));
                FNS_CHECK(cudaMemcpy(p + gb, u, ub, cudaMemcpyHostToDevice));
                FNS_CHECK(cudaMemcpy(p + gb + ub, d, db, cudaMemcpyHostToDevice));
                placement_.vram_bytes += gb + ub + db;
                ++placement_.vram_experts;
            } else {
                char * h = host + (i_host++) * slot[il];
                std::memcpy(h, g, gb);
                std::memcpy(h + gb, u, ub);
                std::memcpy(h + gb + ub, d, db);
                void * dp = nullptr;
                FNS_CHECK(cudaHostGetDevicePointer(&dp, h, 0));
                p = (char *) dp;
                placement_.host_bytes += gb + ub + db;
                ++placement_.host_experts;
            }
            host_table[(size_t) il * NE + e] = p;
            if (!placement_.resident[(size_t) il * NE + e]) slot_host_[(size_t) il * NE + e] = host + (i_host - 1) * slot[il];
        }
        L.exp_table = tables + (size_t) il * NE;
    }
    FNS_CHECK(cudaMemcpy(tables, host_table.data(), host_table.size() * sizeof(char *), cudaMemcpyHostToDevice));
    slot_ptr_ = std::move(host_table);
    weight_bytes_ += placement_.vram_bytes;
    if (opt_.rebalance_every > 0) {
        // Profile scaled to one half-life: a prior, quickly overtaken by the observed routing.
        for (int il = 0; il < NL; ++il) {
            double s = 0;
            for (int e = 0; e < NE; ++e) s += freq_[(size_t) il * NE + e];
            const double k = s > 0 ? hp.n_expert_used * (double) opt_.rebalance_half_life / s : 0.0;
            for (int e = 0; e < NE; ++e) freq_[(size_t) il * NE + e] *= k;
        }
        if (!opt_.rebalance_fused_swap) {
            FNS_CHECK(cudaMalloc(&swap_buf_, *std::max_element(slot.begin(), slot.end())));
            allocs_.push_back(swap_buf_);
        }
    }
}

// Adaptive cache: fold counters into decaying frequencies and exchange slots when routing favors a non-resident expert.
// Copy or fused swaps and the pointer-table uploads are ordered on st_; the next forward sees complete weights.
int Engine::maybe_rebalance(int tokens) {
    if (opt_.rebalance_every <= 0) return 0;
    since_rebalance_ += tokens;
    if (since_rebalance_ < opt_.rebalance_every) return 0;
    const Hparams & hp = m_.hp;
    const int NE = hp.n_expert, NL = (int) m_.layers.size();
    const std::vector<uint32_t> c = routing_counts();
    reset_routing_counts();
    const double decay = std::pow(0.5, (double) since_rebalance_ / opt_.rebalance_half_life);
    since_rebalance_ = 0;
    int swaps = 0;
    for (int il = 0; il < NL; ++il) {
        if (!slot_bytes_[il]) continue;
        double * f = freq_.data() + (size_t) il * NE;
        uint8_t * res = placement_.resident.data() + (size_t) il * NE;
        for (int e = 0; e < NE; ++e) f[e] = f[e] * decay + c[(size_t) il * NE + e];
        bool changed = false;
        for (int k = 0; k < opt_.rebalance_max; ++k) {
            int in = -1, out = -1;
            for (int e = 0; e < NE; ++e) {
                if (res[e]) { if (out < 0 || f[e] < f[out]) out = e; }
                else if (in < 0 || f[e] > f[in]) in = e;
            }
            if (in < 0 || out < 0 || f[in] <= f[out] + 1.0) break;
            const size_t sb = slot_bytes_[il], ii = (size_t) il * NE + in, io = (size_t) il * NE + out;
            char * vram = slot_ptr_[io];           // VRAM slot of the victim
            char * host = slot_host_[ii];          // host slot of the promoted expert
            char * host_dev = slot_ptr_[ii];       // GPU view of the host slot
            if (rebalance_start_ && swaps == 0) FNS_CHECK(cudaEventRecord(rebalance_start_, st_));
            if (opt_.rebalance_fused_swap) {
                k::swap_bytes(vram, host_dev, sb, st_);
            } else {
                FNS_CHECK(cudaMemcpyAsync(swap_buf_, vram, sb, cudaMemcpyDeviceToDevice, st_));
                FNS_CHECK(cudaMemcpyAsync(vram, host, sb, cudaMemcpyHostToDevice, st_));
                FNS_CHECK(cudaMemcpyAsync(host, swap_buf_, sb, cudaMemcpyDeviceToHost, st_));
            }
            if (rebalance_start_) placement_.rebalance_bytes += (uint64_t) sb * 2;
            slot_ptr_[ii] = vram;
            slot_ptr_[io] = host_dev;
            slot_host_[io] = host;
            slot_host_[ii] = nullptr;
            res[in] = 1;
            res[out] = 0;
            changed = true;
            ++swaps;
        }
        if (changed)
            FNS_CHECK(cudaMemcpyAsync(exp_tables_ + (size_t) il * NE, slot_ptr_.data() + (size_t) il * NE,
                                      NE * sizeof(char *), cudaMemcpyHostToDevice, st_));
    }
    // The table copy reads slot_ptr_ (pageable host memory): wait before a later call modifies it.
    if (swaps) {
        if (rebalance_end_) FNS_CHECK(cudaEventRecord(rebalance_end_, st_));
        FNS_CHECK(cudaStreamSynchronize(st_));
        if (rebalance_end_) {
            float ms = 0;
            FNS_CHECK(cudaEventElapsedTime(&ms, rebalance_start_, rebalance_end_));
            placement_.rebalance_ms += ms;
            ++placement_.rebalance_calls;
        }
    }
    placement_.swaps += swaps;
    return swaps;
}

std::vector<uint32_t> Engine::routing_counts() {
    std::vector<uint32_t> c(route_counts_ ? m_.layers.size() * m_.hp.n_expert : 0);
    if (route_counts_) FNS_CHECK(cudaMemcpy(c.data(), route_counts_, c.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return c;
}

void Engine::reset_routing_counts() {
    if (route_counts_) FNS_CHECK(cudaMemset(route_counts_, 0, m_.layers.size() * m_.hp.n_expert * sizeof(uint32_t)));
}

Engine::~Engine() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    for (cudaGraphExec_t e : verify_exec_)
        if (e) cudaGraphExecDestroy(e);
    if (vio_host_) cudaFreeHost(vio_host_);
    if (pick_ev_) cudaEventDestroy(pick_ev_);
    if (rebalance_start_) cudaEventDestroy(rebalance_start_);
    if (rebalance_end_) cudaEventDestroy(rebalance_end_);
    if (cublas_) cublasDestroy(cublas_);
    if (mask_pinned_) cudaFreeHost(mask_pinned_);
    if (vmask_pinned_) cudaFreeHost(vmask_pinned_);
    if (io_host_) cudaFreeHost(io_host_);
    for (void * p : allocs_) cudaFree(p);
    for (void * p : host_allocs_) cudaFreeHost(p);
    if (st_) cudaStreamDestroy(st_);
}

void Engine::gemv(const DeviceMat & w, const float * x, float * y, int64_t ncols) {
    quantize_q8_1(x, xq_, ncols, cols_, st_);
    mmvq(w.type, w.d, ncols, w.ne[1], xq_, y, cols_, st_);
}

// Diagnostic: FNS_TRACE=<layer> prints the first 6 values of each step of that layer (named like llama.cpp nodes,
// to compare with fns-dump and FNS_DUMP_NODE).
void Engine::trace(const char * name, const float * d) {
    if (trace_layer_ != cur_layer_) return;
    float v[6];
    FNS_CHECK(cudaMemcpyAsync(v, d, sizeof(v), cudaMemcpyDeviceToHost, st_));
    FNS_CHECK(cudaStreamSynchronize(st_));
    std::fprintf(stderr, "trace %s-%d: %.6g %.6g %.6g %.6g %.6g %.6g\n", name, cur_layer_, v[0], v[1], v[2], v[3], v[4],
                 v[5]);
}

// One token (cols_ = 1): the reference-path launches. W columns (verify): the same operations per column; dense GEMVs
// use ggml's MMVQ with W columns (ncols_dst = W), per-token expert tables keep their single-token kernel on a grid
// of W * 6 channels.
void Engine::attention_block(const DeviceLayer & L, bool swa) {
    const int32_t * pos = pos_p_;
    const Hparams & hp = m_.hp;
    const int E = hp.n_embd, D = hp.head_dim, W = cols_;
    const int QKV = hp.n_head * D + 2 * hp.n_head_kv * D;
    k::rms_norm_q8_1(x_, (const float *) L.attn_norm.d, h_, xq_, E, hp.rms_eps, W, st_);
    trace("attn_norm", h_);
    mmvq(L.wqkv.type, L.wqkv.d, E, L.wqkv.ne[1], xq_, q_, W, st_);
    trace("Qcur", q_);
    trace("Kcur", k_);
    trace("Vcur", v_);
    // With FNS_TRACE: three-kernel path to expose Qrope and Krope; otherwise one launch, same arithmetic.
    if (trace_layer_ == cur_layer_ && W == 1) {
        k::rms_norm_mul(q_, (const float *) L.q_norm.d, q_, D, hp.n_head, hp.rms_eps, st_);
        k::rms_norm_mul(k_, (const float *) L.k_norm.d, k_, D, hp.n_head_kv, hp.rms_eps, st_);
        if (swa) {
            k::rope_neox(q_, hp.n_head, D, pos, hp.rope_base, st_);
            k::rope_neox(k_, hp.n_head_kv, D, pos, hp.rope_base, st_);
        }
        trace("Qrope", q_);
        trace("Krope", k_);
        if (L.kv_q8) throw std::runtime_error("FNS_TRACE: global Q8_0 KV is not supported");
        k::store_kv(k_, v_, L.k_cache, L.v_cache, pos, L.slots, hp.n_head_kv * D, st_);
    } else {
        k::qk_norm_rope_store(q_, k_, v_, (const float *) L.q_norm.d, (const float *) L.k_norm.d, L.k_cache, L.v_cache,
                              pos, hp.n_head, hp.n_head_kv, D, L.slots, hp.rms_eps, hp.rope_base, swa, W, QKV, st_,
                              L.kv_q8);
    }
    // llama.cpp window (llama-hparams.h:512-517): position p0 is masked if pos - p0 >= n_swa.
    k::attention(q_, L.k_cache, L.v_cache, (const float *) L.sinks.d, att_, scores_, hp.n_head, hp.n_head_kv, D, pos,
                 swa ? hp.n_swa : 0, L.slots, opt_.n_ctx, 1.0f / std::sqrt((float) D), W, QKV, st_, L.kv_q8,
                 opt_.attention_mma && !swa);
    trace("kqv_out", att_);
    gemv(L.wo, att_, o_, (int64_t) hp.n_head * D);
    k::add_inplace(x_, o_, W * E, st_);  // ffn_inp = attn + inpSA
    trace("ffn_inp", x_);
}

// GLU of two dense GEMVs: ggml's fused launch for one token. At W columns (ggml reserves GLU fusion for ncols_dst = 1),
// two MMVQs then the fusion epilogue, same sums and same per-column operations.
static void dense_glu(const DeviceMat & g, const DeviceMat & u, int64_t ncols, int64_t nrows, const void * xq,
                      float * tmp, float * act, int W, cudaStream_t st) {
    if (W == 1) {
        mmvq_swiglu(g.type, g.d, u.d, ncols, nrows, xq, act, st);
        return;
    }
    mmvq(g.type, g.d, ncols, nrows, xq, tmp, W, st);
    mmvq(u.type, u.d, ncols, nrows, xq, act, W, st);
    k::glu_mmvq(tmp, act, act, (int) (W * nrows), st);
}

void Engine::ffn_block(const DeviceLayer & L, bool dense) {
    const Hparams & hp = m_.hp;
    const int E = hp.n_embd, W = cols_;
    k::rms_norm_q8_1(x_, (const float *) L.ffn_norm.d, h_, xq_, E, hp.rms_eps, W, st_);
    trace("ffn_norm", h_);
    if (dense) {
        dense_glu(L.ffn_gate, L.ffn_up, E, hp.n_ff, xq_, gate_, act_, W, st_);
        trace("ffn_swiglu", act_);
        gemv(L.ffn_down, act_, ffn_, hp.n_ff);
        trace("ffn_out", ffn_);
        k::add_inplace(x_, ffn_, W * E, st_);
        trace("l_out", x_);
        return;
    }
    const int K = hp.n_expert_used, F = hp.n_ff_exp;
    // Router BF16 to F32, sigmoid, top-6, normalization (llama-graph.cpp build_moe_ffn).
    k::gemv_bf16(L.gate_inp.d, h_, router_, E, hp.n_expert, W, st_);
    trace("ffn_moe_logits", router_);
    if (W == 1) k::route_topk(router_, ids_, wsel_, hp.n_expert, K, st_);
    else k::route_topk_batch(router_, ids_, wsel_, hp.n_expert, K, W, st_);
    trace("ffn_moe_weights_norm", wsel_);
    if (route_counts_) {
        for (int c = 0; c < W; ++c)
            k::count_routes(ids_ + (size_t) c * K, route_counts_ + (size_t) cur_layer_ * hp.n_expert, K, st_);
    }
    // Experts: gate, up and SwiGLU in one launch (one input row per column), F16 guard, down (six input rows per
    // column). The shared expert (FFN 1024) does not have the routed experts' shape (512), so ggml's "shared" fusion
    // does not apply: it gets its own launch.
    // xq_ still holds the quantization of ffn_norm (rms_norm_q8_1): the router only reads h_.
    mmvq_experts_swiglu(L.gu_type, L.exp_table, 0, L.gate_bytes, E, F, xq_, W, ids_, W * K, act_, st_);
    dense_glu(L.gate_shexp, L.up_shexp, E, hp.n_ff_shexp, xq_, gate_, up_, W, st_);
    trace("ffn_moe_swiglu", act_);
    trace("ffn_swiglu", up_);
    k::amax_guard_q8_1(act_, escale_, xq_, F, W * K, st_);
    mmvq_experts(L.down_type, L.exp_table, L.gate_bytes + L.up_bytes, F, E, xq_, W * K, ids_, W * K, dexp_, st_);
    trace("ffn_moe_down_scaled_by_1/e", dexp_);
    gemv(L.down_shexp, up_, sh_, hp.n_ff_shexp);
    trace("ffn_shexp", sh_);
    if (W == 1) k::moe_combine(dexp_, escale_, wsel_, sh_, x_, E, K, st_);
    else k::moe_combine_batch(dexp_, escale_, wsel_, sh_, x_, E, K, W, st_);
    trace("l_out", x_);
}

// The forward of cols_ columns, with no host synchronization: capturable in a CUDA graph.
void Engine::forward() {
    const Hparams & hp = m_.hp;
    k::embed_q8_0(tok_embd_.d, tok_p_, x_, hp.n_embd, cols_, st_);
    for (size_t il = 0; il < layers_.size(); ++il) {
        cur_layer_ = (int) il;
        attention_block(layers_[il], m_.layers[il].swa);
        ffn_block(layers_[il], m_.layers[il].dense);
        if (layer_buf_) {
            FNS_CHECK(cudaMemcpyAsync(layer_buf_ + il * hp.n_embd, x_, (size_t) hp.n_embd * sizeof(float),
                                      cudaMemcpyDeviceToHost, st_));
        }
    }
    k::rms_norm_mul(x_, (const float *) output_norm_.d, h_, hp.n_embd, cols_, hp.rms_eps, st_);
    gemv(output_, h_, logits_, hp.n_embd);
}

void Engine::verify(const int32_t * ids, int W, int p0, int32_t * out, float * logits, const uint32_t * masks,
                    const SampleParams & sp, uint64_t counter) {
    const Hparams & hp = m_.hp;
    if (W < 1 || W > opt_.verify_cols) throw std::runtime_error("verify: W outside [1, verify_cols]");
    if (p0 < 0 || p0 + W > opt_.n_ctx) throw std::runtime_error("verify: positions outside the context");
    vio_host_[0] = p0;
    for (int c = 0; c < W; ++c) {
        if (ids[c] < 0 || ids[c] >= hp.n_vocab) throw std::runtime_error("verify: token outside the vocabulary");
        vio_host_[1 + c] = ids[c];
    }
    FNS_CHECK(cudaMemcpyAsync(vio_, vio_host_, (1 + W) * sizeof(int32_t), cudaMemcpyHostToDevice, st_));
    cols_ = W;
    tok_p_ = vio_ + 1;
    pos_p_ = vio_;
    run_forward(true, verify_exec_[W]);
    cols_ = 1;
    tok_p_ = io_;
    pos_p_ = io_ + 1;
    const size_t words = (size_t) (hp.n_vocab + 31) / 32;
    if (masks) {
        if (!vmask_dev_) {
            FNS_CHECK(cudaMalloc(&vmask_dev_, (size_t) opt_.verify_cols * words * sizeof(uint32_t)));
            allocs_.push_back(vmask_dev_);
            FNS_CHECK(cudaMallocHost(&vmask_pinned_, (size_t) opt_.verify_cols * words * sizeof(uint32_t)));
        }
        // Synchronous call: the previous copy from vmask_pinned_ has completed.
        std::memcpy(vmask_pinned_, masks, W * words * sizeof(uint32_t));
        FNS_CHECK(cudaMemcpyAsync(vmask_dev_, vmask_pinned_, W * words * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                  st_));
    }
    // Choice of each column: the same sampler as pick() (argmax, or top-k / top-p / temperature), Philox counter
    // counter + c. With logits bit-identical to the sequential path, column c draws the token the sequential path would
    // draw at the same generation index. Presence penalty: column c sees seen_ plus ids[1 .. c], the tokens the
    // sequential path would have chosen before it if the window is committed up to there (presence_commit afterwards).
    if (sp.presence_penalty != 0.0f && !seen_) reset_presence();
    for (int c = 0; c < W; ++c) {
        const uint32_t * m = masks ? vmask_dev_ + c * words : nullptr;
        float * lg = logits_ + (size_t) c * hp.n_vocab;
        if (sp.presence_penalty != 0.0f)
            k::presence_apply_col(lg, seen_, vio_ + 2, c, hp.n_vocab, sp.presence_penalty, st_);
        if (sp.temperature <= 0.0f) k::argmax(lg, hp.n_vocab, vio_ + 9 + c, st_, m);
        else k::sample(lg, hp.n_vocab, m, sp.top_k, sp.top_p, sp.temperature, sp.seed, counter + c, vio_ + 9 + c, st_);
    }
    FNS_CHECK(cudaMemcpyAsync(vio_host_ + 9, vio_ + 9, W * sizeof(int32_t), cudaMemcpyDeviceToHost, st_));
    if (logits) {
        FNS_CHECK(cudaMemcpyAsync(logits, logits_, (size_t) W * hp.n_vocab * sizeof(float), cudaMemcpyDeviceToHost,
                                  st_));
    }
    FNS_CHECK(cudaStreamSynchronize(st_));
    for (int c = 0; c < W; ++c) out[c] = vio_host_[9 + c];
}

void Engine::launch_step(int32_t token, int pos, float * layer_out) {
    const Hparams & hp = m_.hp;
    if (pos < 0 || pos >= opt_.n_ctx) throw std::runtime_error("position outside the context");
    if (token < 0 || token >= hp.n_vocab) throw std::runtime_error("token outside the vocabulary");
    io_host_[0] = token;
    io_host_[1] = pos;
    FNS_CHECK(cudaMemcpyAsync(io_, io_host_, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, st_));
    // The layer dump copies to a host buffer that changes between calls: direct path, no graph.
    layer_buf_ = layer_out;
    run_forward(layer_out == nullptr, graph_exec_);
    layer_buf_ = nullptr;
}

void Engine::run_forward(bool graph, cudaGraphExec_t & exec) {
    if (!use_graph_ || !graph) {
        forward();
        return;
    }
    if (!exec) {
        cudaGraph_t g = nullptr;
        FNS_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
        forward();
        FNS_CHECK(cudaStreamEndCapture(st_, &g));
        FNS_CHECK(cudaGraphInstantiate(&exec, g, 0));
        FNS_CHECK(cudaGraphDestroy(g));
    }
    FNS_CHECK(cudaGraphLaunch(exec, st_));
}

void Engine::step_next(int pos) {
    if (pos < 0 || pos >= opt_.n_ctx) throw std::runtime_error("position outside the context");
    k::io_advance(io_, pos, st_);
    run_forward(true, graph_exec_);
}

void Engine::step(int32_t token, int pos, float * logits, float * layer_out) {
    launch_step(token, pos, layer_out);
    if (logits) {
        FNS_CHECK(cudaMemcpyAsync(logits, logits_, (size_t) m_.hp.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, st_));
    } else {
        k::argmax(logits_, m_.hp.n_vocab, io_ + 2, st_);
        FNS_CHECK(cudaMemcpyAsync(io_host_ + 2, io_ + 2, sizeof(int32_t), cudaMemcpyDeviceToHost, st_));
    }
    FNS_CHECK(cudaStreamSynchronize(st_));
}

int32_t Engine::step_greedy(int32_t token, int pos) {
    step(token, pos, nullptr);
    return io_host_[2];
}

int32_t Engine::pick(const uint32_t * mask_host, const SampleParams & sp, uint64_t counter) {
    pick_async(mask_host, sp, counter);
    return pick_wait();
}

void Engine::pick_async(const uint32_t * mask_host, const SampleParams & sp, uint64_t counter) {
    const Hparams & hp = m_.hp;
    const size_t words = (size_t) (hp.n_vocab + 31) / 32;
    if (mask_host) {
        if (!mask_dev_) {
            FNS_CHECK(cudaMalloc(&mask_dev_, words * sizeof(uint32_t)));
            allocs_.push_back(mask_dev_);
            FNS_CHECK(cudaMallocHost(&mask_pinned_, words * sizeof(uint32_t)));
        }
        // mask_pinned_ is rewritten only after pick_wait() of the previous pick: its copy has then completed.
        std::memcpy(mask_pinned_, mask_host, words * sizeof(uint32_t));
        FNS_CHECK(cudaMemcpyAsync(mask_dev_, mask_pinned_, words * sizeof(uint32_t), cudaMemcpyHostToDevice, st_));
    }
    const uint32_t * mask = mask_host ? mask_dev_ : nullptr;
    if (sp.presence_penalty != 0.0f) {
        if (!seen_) reset_presence();
        k::presence_apply(logits_, seen_, hp.n_vocab, sp.presence_penalty, st_);
    }
    if (sp.temperature <= 0.0f) k::argmax(logits_, hp.n_vocab, io_ + 2, st_, mask);
    else k::sample(logits_, hp.n_vocab, mask, sp.top_k, sp.top_p, sp.temperature, sp.seed, counter, io_ + 2, st_);
    if (sp.presence_penalty != 0.0f) k::presence_mark(seen_, io_ + 2, st_);
    FNS_CHECK(cudaMemcpyAsync(io_host_ + 2, io_ + 2, sizeof(int32_t), cudaMemcpyDeviceToHost, st_));
    if (!pick_ev_) FNS_CHECK(cudaEventCreateWithFlags(&pick_ev_, cudaEventDisableTiming));
    FNS_CHECK(cudaEventRecord(pick_ev_, st_));
}

int32_t Engine::pick_wait() {
    FNS_CHECK(cudaEventSynchronize(pick_ev_));
    return io_host_[2];
}

void Engine::reset_presence() {
    if (!seen_) {
        FNS_CHECK(cudaMalloc(&seen_, (size_t) m_.hp.n_vocab * sizeof(float)));
        allocs_.push_back(seen_);
    }
    FNS_CHECK(cudaMemsetAsync(seen_, 0, (size_t) m_.hp.n_vocab * sizeof(float), st_));
}

void Engine::presence_commit(int n) {
    if (n < 0 || n > opt_.verify_cols) throw std::runtime_error("presence_commit: n outside [0, verify_cols]");
    if (!seen_) reset_presence();
    if (n > 0) k::presence_mark_n(seen_, vio_ + 9, n, st_);
}

}  // namespace fns
