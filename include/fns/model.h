// include/fns/model.h - Flint GGUF file: checked hyperparameters and tensors, without copying (mmap).
#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace fns {

// A view of a tensor in the mapped file. `ne` in ggml order (ne[0] = contiguous dimension).
struct Tensor {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[4] = {1, 1, 1, 1};
    const void * data = nullptr;  // Points into the mapped file.
    size_t nbytes = 0;
};

struct Layer {
    bool swa = false;    // sliding-window layer (RoPE + window of 513)
    bool dense = false;  // layers 0-1: dense FFN of 6144
    Tensor attn_norm, wq, wk, wv, wo, q_norm, k_norm, sinks, ffn_norm;
    Tensor ffn_gate, ffn_up, ffn_down;                    // dense
    Tensor gate_inp, gate_exps, up_exps, down_exps;       // MoE
    Tensor gate_shexp, up_shexp, down_shexp;              // shared expert
};

struct Hparams {
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0;
    int n_ff = 0, n_ff_exp = 0, n_ff_shexp = 0, n_expert = 0, n_expert_used = 0, n_dense_lead = 0;
    int n_swa = 0, n_vocab = 0, n_ctx_train = 0;
    float rms_eps = 0.0f, rope_base = 0.0f, expert_weights_scale = 1.0f;
    bool expert_weights_norm = false;
};

class Model {
public:
    explicit Model(const std::string & path);  // throws std::runtime_error if the file is not a valid Flint model
    ~Model();
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    Hparams hp;
    Tensor tok_embd, output_norm, output;
    std::vector<Layer> layers;
    std::vector<int32_t> eog_ids;

private:
    const Tensor & find(const std::string & name) const;
    Tensor load(const std::string & name);
    void verify();

    gguf_context * ctx_ = nullptr;
    void * file_ = nullptr;
    void * mapping_ = nullptr;
    const uint8_t * base_ = nullptr;
    size_t size_ = 0;
    std::vector<Tensor> all_;
};

}  // namespace fns
