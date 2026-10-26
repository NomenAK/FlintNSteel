// src/model.cpp - GGUF loader for Flint: metadata via the ggml-base gguf API, tensor data via Win32 file mapping.
#include "fns/model.h"

#include "fns/ggml_kernels.h"
#include "gguf.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstring>
#include <stdexcept>

namespace fns {

namespace {

[[noreturn]] void fail(const std::string & msg) { throw std::runtime_error("model: " + msg); }

int64_t key(const gguf_context * ctx, const char * name) {
    const int64_t id = gguf_find_key(ctx, name);
    if (id < 0) fail(std::string("missing key: ") + name);
    return id;
}

uint32_t u32(const gguf_context * ctx, const char * name) { return gguf_get_val_u32(ctx, key(ctx, name)); }
float f32(const gguf_context * ctx, const char * name) { return gguf_get_val_f32(ctx, key(ctx, name)); }

}  // namespace

Model::Model(const std::string & path) {
    gguf_init_params params = {/*no_alloc =*/ true, /*ctx =*/ nullptr};
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) fail("cannot read GGUF header of " + path);

    if (std::strcmp(gguf_get_val_str(ctx_, key(ctx_, "general.architecture")), "flint") != 0) {
        fail("general.architecture is not 'flint'");
    }

    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f == INVALID_HANDLE_VALUE) fail("cannot open: " + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(f, &sz);
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) fail("CreateFileMapping failed");
    base_ = (const uint8_t *) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!base_) fail("MapViewOfFile failed");
    file_ = f;
    mapping_ = m;
    size_ = (size_t) sz.QuadPart;

    const size_t data_off = gguf_get_data_offset(ctx_);
    const int64_t n = gguf_get_n_tensors(ctx_);
    all_.reserve((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        Tensor t;
        t.name = gguf_get_tensor_name(ctx_, i);
        t.type = gguf_get_tensor_type(ctx_, i);
        const int64_t * ne = gguf_get_tensor_ne(ctx_, i);
        for (int d = 0; d < 4; ++d) t.ne[d] = ne[d];
        t.nbytes = gguf_get_tensor_size(ctx_, i);
        const size_t off = data_off + gguf_get_tensor_offset(ctx_, i);
        if (off + t.nbytes > size_) fail("tensor outside the file: " + t.name);
        t.data = base_ + off;
        all_.push_back(std::move(t));
    }

    hp.n_layer = (int) u32(ctx_, "flint.block_count");
    hp.n_embd = (int) u32(ctx_, "flint.embedding_length");
    hp.n_ff = (int) u32(ctx_, "flint.feed_forward_length");
    hp.n_head = (int) u32(ctx_, "flint.attention.head_count");
    hp.n_head_kv = (int) u32(ctx_, "flint.attention.head_count_kv");
    hp.head_dim = (int) u32(ctx_, "flint.attention.key_length");
    hp.n_expert = (int) u32(ctx_, "flint.expert_count");
    hp.n_expert_used = (int) u32(ctx_, "flint.expert_used_count");
    hp.n_ff_exp = (int) u32(ctx_, "flint.expert_feed_forward_length");
    hp.n_ff_shexp = (int) u32(ctx_, "flint.expert_shared_feed_forward_length");
    hp.n_dense_lead = (int) u32(ctx_, "flint.leading_dense_block_count");
    hp.n_swa = (int) u32(ctx_, "flint.attention.sliding_window");
    hp.n_ctx_train = (int) u32(ctx_, "flint.context_length");
    hp.rms_eps = f32(ctx_, "flint.attention.layer_norm_rms_epsilon");
    hp.rope_base = f32(ctx_, "flint.rope.freq_base");
    hp.expert_weights_scale = f32(ctx_, "flint.expert_weights_scale");
    hp.expert_weights_norm = gguf_get_val_bool(ctx_, key(ctx_, "flint.expert_weights_norm"));

    const int64_t swa_id = key(ctx_, "flint.attention.sliding_window_pattern");
    const int64_t rope_id = key(ctx_, "flint.attention.rope_pattern");
    if ((int) gguf_get_arr_n(ctx_, swa_id) != hp.n_layer || (int) gguf_get_arr_n(ctx_, rope_id) != hp.n_layer) {
        fail("SWA / RoPE patterns differ in length from the layer count");
    }
    const int8_t * swa = (const int8_t *) gguf_get_arr_data(ctx_, swa_id);
    const int8_t * rope = (const int8_t *) gguf_get_arr_data(ctx_, rope_id);

    const int64_t eog_id = key(ctx_, "tokenizer.ggml.eog_token_ids");
    const int32_t * eog = (const int32_t *) gguf_get_arr_data(ctx_, eog_id);
    eog_ids.assign(eog, eog + gguf_get_arr_n(ctx_, eog_id));

    tok_embd = load("token_embd.weight");
    output_norm = load("output_norm.weight");
    output = load("output.weight");
    hp.n_vocab = (int) output.ne[1];

    layers.resize((size_t) hp.n_layer);
    for (int il = 0; il < hp.n_layer; ++il) {
        Layer & L = layers[(size_t) il];
        if ((swa[il] != 0) != (rope[il] != 0)) {
            fail("layer " + std::to_string(il) + ": RoPE and sliding window should coincide");
        }
        L.swa = swa[il] != 0;
        L.dense = il < hp.n_dense_lead;
        const std::string p = "blk." + std::to_string(il) + ".";
        L.attn_norm = load(p + "attn_norm.weight");
        L.wq = load(p + "attn_q.weight");
        L.wk = load(p + "attn_k.weight");
        L.wv = load(p + "attn_v.weight");
        L.wo = load(p + "attn_output.weight");
        L.q_norm = load(p + "attn_q_norm.weight");
        L.k_norm = load(p + "attn_k_norm.weight");
        L.sinks = load(p + "attn_sinks.weight");
        L.ffn_norm = load(p + "ffn_norm.weight");
        if (L.dense) {
            L.ffn_gate = load(p + "ffn_gate.weight");
            L.ffn_up = load(p + "ffn_up.weight");
            L.ffn_down = load(p + "ffn_down.weight");
        } else {
            L.gate_inp = load(p + "ffn_gate_inp.weight");
            L.gate_exps = load(p + "ffn_gate_exps.weight");
            L.up_exps = load(p + "ffn_up_exps.weight");
            L.down_exps = load(p + "ffn_down_exps.weight");
            L.gate_shexp = load(p + "ffn_gate_shexp.weight");
            L.up_shexp = load(p + "ffn_up_shexp.weight");
            L.down_shexp = load(p + "ffn_down_shexp.weight");
        }
    }
    verify();
}

Model::~Model() {
    if (base_) UnmapViewOfFile(base_);
    if (mapping_) CloseHandle((HANDLE) mapping_);
    if (file_) CloseHandle((HANDLE) file_);
    if (ctx_) gguf_free(ctx_);
}

const Tensor & Model::find(const std::string & name) const {
    for (const Tensor & t : all_) {
        if (t.name == name) return t;
    }
    fail("missing tensor: " + name);
}

Tensor Model::load(const std::string & name) { return find(name); }

// Fixed hyperparameters of this model: the engine is written for this shape and rejects any other.
void Model::verify() {
    auto expect = [](bool ok, const char * what) {
        if (!ok) fail(std::string("unexpected hyperparameter: ") + what);
    };
    expect(hp.n_layer == 49, "block_count == 49");
    expect(hp.n_embd == 2048, "embedding_length == 2048");
    expect(hp.n_head == 32 && hp.n_head_kv == 4 && hp.head_dim == 128, "GQA 32/4, head_dim 128");
    expect(hp.n_expert == 256 && hp.n_expert_used == 6, "256 experts, top-6");
    expect(hp.n_ff == 6144 && hp.n_ff_exp == 512 && hp.n_ff_shexp == 1024, "FFN 6144 / 512 / 1024");
    expect(hp.n_dense_lead == 2, "2 dense layers");
    expect(hp.n_swa == 513, "sliding window of 513");
    expect(hp.n_vocab == 150144, "vocabulary 150144");
    expect(hp.expert_weights_norm && hp.expert_weights_scale == 1.0f, "expert weights normalized, scale 1");

    auto check = [](const Tensor & t, ggml_type want_or_any, std::initializer_list<int64_t> ne) {
        int d = 0;
        for (int64_t v : ne) {
            if (t.ne[d++] != v) fail("unexpected shape: " + t.name);
        }
        if (want_or_any != GGML_TYPE_COUNT && t.type != want_or_any) fail("unexpected type: " + t.name);
        if (want_or_any == GGML_TYPE_COUNT && !mmvq_supports(t.type)) {
            fail("unsupported format (" + std::string(ggml_type_name(t.type)) + "): " + t.name);
        }
    };
    const int64_t E = hp.n_embd, Q = (int64_t) hp.n_head * hp.head_dim, KV = (int64_t) hp.n_head_kv * hp.head_dim;
    check(tok_embd, GGML_TYPE_Q8_0, {E, hp.n_vocab});
    check(output, GGML_TYPE_COUNT, {E, hp.n_vocab});
    check(output_norm, GGML_TYPE_F32, {E});
    for (const Layer & L : layers) {
        check(L.attn_norm, GGML_TYPE_F32, {E});
        check(L.wq, GGML_TYPE_COUNT, {E, Q});
        check(L.wk, GGML_TYPE_COUNT, {E, KV});
        check(L.wv, GGML_TYPE_COUNT, {E, KV});
        check(L.wo, GGML_TYPE_COUNT, {Q, E});
        check(L.q_norm, GGML_TYPE_F32, {hp.head_dim});
        check(L.k_norm, GGML_TYPE_F32, {hp.head_dim});
        check(L.sinks, GGML_TYPE_F32, {hp.n_head});
        check(L.ffn_norm, GGML_TYPE_F32, {E});
        if (L.dense) {
            check(L.ffn_gate, GGML_TYPE_COUNT, {E, hp.n_ff});
            check(L.ffn_up, GGML_TYPE_COUNT, {E, hp.n_ff});
            check(L.ffn_down, GGML_TYPE_COUNT, {hp.n_ff, E});
        } else {
            check(L.gate_inp, GGML_TYPE_BF16, {E, hp.n_expert});
            check(L.gate_exps, GGML_TYPE_COUNT, {E, hp.n_ff_exp, hp.n_expert});
            check(L.up_exps, GGML_TYPE_COUNT, {E, hp.n_ff_exp, hp.n_expert});
            check(L.down_exps, GGML_TYPE_COUNT, {hp.n_ff_exp, E, hp.n_expert});
            check(L.gate_shexp, GGML_TYPE_COUNT, {E, hp.n_ff_shexp});
            check(L.up_shexp, GGML_TYPE_COUNT, {E, hp.n_ff_shexp});
            check(L.down_shexp, GGML_TYPE_COUNT, {hp.n_ff_shexp, E});
        }
    }
}

}  // namespace fns
