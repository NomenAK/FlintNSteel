// include/fns/engine.h - reference engine: one token at a time, everything in VRAM, F32 activations.
#pragma once

#include "fns/ggml_kernels.h"
#include "fns/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

struct cublasContext;  // cublasHandle_t (cublas_api.h); keeps cuBLAS out of the engine's public users

namespace fns {

struct DeviceMat {
    ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[4] = {1, 1, 1, 1};
    void * d = nullptr;
};

struct DeviceLayer {
    DeviceMat attn_norm, wqkv, wo, q_norm, k_norm, sinks, ffn_norm;  // wqkv: Q, K, V concatenated row-wise
    DeviceMat ffn_gate, ffn_up, ffn_down;
    DeviceMat gate_inp;
    DeviceMat gate_shexp, up_shexp, down_shexp;
    // Routed experts: expert e is the slot [gate | up | down] starting at exp_table[e], either in VRAM or as a copy in
    // pinned host memory read by the GPU. exp_table is in GPU memory.
    ggml_type gu_type = GGML_TYPE_COUNT, down_type = GGML_TYPE_COUNT;
    size_t gate_bytes = 0, up_bytes = 0;  // per expert: up at +gate_bytes, down at +gate_bytes+up_bytes
    const char * const * exp_table = nullptr;
    void * k_cache = nullptr;  // half [slots][n_head_kv][head_dim], or Q8_0 [slots][n_head_kv][4 blocks] (kv_q8)
    void * v_cache = nullptr;
    int slots = 0;             // n_ctx (global) or R (sliding ring)
    bool kv_q8 = false;        // global layer with EngineOptions::kv_global_q8
};

struct EngineOptions {
    int n_ctx = 4096;  // max positions of one sequence
    int ring = 1024;   // ring positions of sliding-window layers (>= 513)
    // Expert cache. VRAM left free after loading goes to experts. Experts that do not fit stay in pinned RAM and are
    // read by the GPU (zero-copy).
    size_t vram_reserve = (size_t) 768 << 20;
    long long expert_vram = -1;          // bytes of experts in VRAM; < 0: everything the reserve leaves free
    std::string expert_profile;          // uint32 routing counters [n_layer][n_expert] (initial placement)
    bool count_routing = false;          // count routed experts per layer (routing_counts)
    // Prefill: tokens per batch (0: no prefill buffers, prefill() goes token by token). The sliding ring must cover
    // ring >= 513 + ubatch - 1; the constructor enlarges it when needed.
    int ubatch = 0;
    // Verify with W columns: maximum columns of verify() (0: no buffers; 1 to 8). The sliding ring is enlarged to
    // 513 + verify_cols - 1 when needed.
    int verify_cols = 0;
    // Global layers' KV in Q8_0 (opt-in, changes the bits). Sliding rings stay F16. Prefill dequantizes the global KV
    // to F16 (buffer of n_ctx positions) before ggml's FlashAttention.
    bool kv_global_q8 = false;
    // INT8 FlashAttention for prefill on global layers (opt-in, changes the bits; src/fa_int8.cu).
    // Incompatible with kv_global_q8.
    bool fa_int8 = false;
    // Prefill FlashAttention for sliding layers, limited to visible tiles (opt-in, changes the bits; src/fa_int8.cu,
    // k_fa_swa). ggml-style MMA f16, with neither mask nor full ring.
    bool fa_swa = false;
    // Global attention with MMA F16->F32 in decode/verify (opt-in, changes the bits).
    bool attention_mma = false;
    // Adaptive expert cache (Q4 hybrid): every `rebalance_every` tokens (0: static), decaying routing counters
    // (half-life `rebalance_half_life` tokens, profile as prior); per layer, at most `rebalance_max` swaps of the most
    // routed non-resident expert with the least routed resident one. Same bits (only weight placement changes).
    // Implies count_routing.
    int rebalance_every = 0;
    int rebalance_half_life = 2048;
    int rebalance_max = 4;
    bool rebalance_fused_swap = false;  // swap VRAM and mapped host slots in place on the GPU
};

// Sampling parameters (Engine::pick). temperature <= 0: greedy.
struct SampleParams {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;  // 0: 64 candidates (kernel limit)
    uint64_t seed = 0;
    // Presence penalty (rec): logits -= presence_penalty for each token already chosen by pick() since the last
    // reset_presence(). 0: none.
    float presence_penalty = 0.0f;
};

// Expert placement summary.
struct ExpertPlacement {
    size_t vram_bytes = 0, host_bytes = 0;
    int vram_experts = 0, host_experts = 0;
    std::vector<uint8_t> resident;  // [n_layer][n_expert]: 1 if in VRAM (dense layers: 0)
    long long swaps = 0;            // VRAM <-> RAM swaps done by maybe_rebalance()
    double rebalance_ms = 0;        // GPU event time for swap and pointer-table work when timing is enabled
    long long rebalance_calls = 0;  // timed calls that performed at least one swap
    uint64_t rebalance_bytes = 0;  // logical bytes exchanged (two slot sizes per swap)
    bool rebalance_timing = false;
};

class Engine {
public:
    Engine(const Model & model, const EngineOptions & opt);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    // Processes `token` at position `pos` (0, 1, 2... without gaps) and copies the logits [n_vocab] into `logits`
    // (if non-null). If `layer_out` is non-null, copies each layer's output into it, [n_layer][n_embd] (diagnostics).
    void step(int32_t token, int pos, float * logits, float * layer_out = nullptr);

    // Like step, but the argmax of the logits is computed on the GPU: only the next token id returns to the host
    // (4 bytes instead of 600 KB). Ties: smallest index, as std::max_element.
    int32_t step_greedy(int32_t token, int pos);

    // Grammar and sampling: forward of one token without a choice, then pick() chooses the next token from the current
    // logits: argmax (temperature <= 0), or top-k / top-p / temperature sampling (Philox, seed sp.seed, counter `counter`),
    // among the tokens allowed by the bitmask (xgrammar, uint32 [(n_vocab + 31) / 32], 1 = allowed; null: all). pick() is
    // also valid after prefill(): it rereads the logits of the last position. Returns -1 if the mask allows no token.
    void step_logits(int32_t token, int pos) { launch_step(token, pos, nullptr); }
    int32_t pick(const uint32_t * mask_host, const SampleParams & sp = {}, uint64_t counter = 0);
    // Chained decode (fns-serve): pick_async() queues the choice and its copy to the host; step_next(pos) immediately
    // queues the forward of the chosen token, read on the GPU (io_advance), without waiting for the host; pick_wait() waits
    // only for the choice. The next forward is already queued when the host receives the token: no gap between two graphs.
    // If the token turns out to be a stop token, that forward wrote the KV of this token at `pos` (valid).
    void pick_async(const uint32_t * mask_host, const SampleParams & sp = {}, uint64_t counter = 0);
    int32_t pick_wait();
    void step_next(int pos);
    // Forgets the tokens seen by the presence penalty (start of each request).
    void reset_presence();

    // Batched prefill: processes ids[0 .. n) at positions p0 .. p0 + n - 1 in batches of at most `ubatch` tokens (MMQ GEMM,
    // FlashAttention), fills the cache, and returns the argmax of the last position's logits. If `logits` is non-null,
    // copies each position's logits into it, [n][n_vocab] (parity checks; slow).
    int32_t prefill(const int32_t * ids, int n, int p0, float * logits = nullptr);

    // Verify with W columns: forward of ids[0 .. W) at positions p0 .. p0 + W - 1 in one pass (MMVQ with W columns, causal
    // attention per column), writes KV for the W positions; out[c] = choice of column c with the sampler of pick() (argmax
    // if sp.temperature <= 0, else top-k / top-p / temperature with Philox sp.seed and counter counter + c), among the
    // tokens allowed by masks[c] (xgrammar bitmask, uint32 [W][(n_vocab + 31) / 32] on the host; null: all). Presence
    // penalty: column c counts ids[1 .. c] as already chosen; after verify, presence_commit(n) marks out[0 .. n) as
    // committed. If `logits` is non-null, copies [W][n_vocab] there. Rejected columns leave KV that the next forward
    // rewrites at the same position (nothing is read beyond each query's position). W <= verify_cols.
    void verify(const int32_t * ids, int W, int p0, int32_t * out, float * logits = nullptr,
                const uint32_t * masks = nullptr, const SampleParams & sp = {}, uint64_t counter = 0);
    void presence_commit(int n);

    size_t weight_bytes() const { return weight_bytes_; }  // VRAM weights (resident experts included)
    const ExpertPlacement & placement() const { return placement_; }
    int ring() const { return opt_.ring; }  // sliding ring positions kept (enlarged for batches)

    // With count_routing: number of selections of each expert since the last reset, [n_layer][n_expert].
    std::vector<uint32_t> routing_counts();
    void reset_routing_counts();
    // Adaptive cache: call between two tokens (fns-serve, fns-bench) with the number of tokens processed since the previous
    // call; acts only once `rebalance_every` tokens have accumulated. Swaps are queued on the engine stream, so they are
    // ordered with the forwards. Returns the number of swaps.
    int maybe_rebalance(int tokens);

    // Diagnostics: FNS_TRACE=<layer> prints the start of each step of that layer (Engine::trace).
    int trace_layer_ = -1;

private:
    DeviceMat upload(const Tensor & t);
    DeviceMat upload_qkv(const Tensor & q, const Tensor & k, const Tensor & v);
    void place_experts();
    // Adaptive cache state: pointer read by the kernel for each expert; host address of non-resident slots.
    std::vector<char *> slot_ptr_;     // [n_layer][n_expert]: VRAM or mapped host memory (GPU view)
    std::vector<char *> slot_host_;    // [n_layer][n_expert]: host address of a non-resident slot (else null)
    std::vector<size_t> slot_bytes_;   // [n_layer]: size of one slot (0: dense layer)
    std::vector<double> freq_;         // [n_layer][n_expert]: decaying counters
    char * swap_buf_ = nullptr;        // VRAM, one slot of the largest layer: swap buffer
    char ** exp_tables_ = nullptr;     // GPU, [n_layer][n_expert]
    int since_rebalance_ = 0;
    cudaEvent_t rebalance_start_ = nullptr, rebalance_end_ = nullptr;
    // INT8 FA of prefill (fa_int8): shared Q buffer, quantized K and mean per global layer.
    void * fa_q_ = nullptr;
    std::vector<void *> fa_k_;
    std::vector<float *> fa_mean_;
    std::vector<int> fa_quantized_;
    void launch_step(int32_t token, int pos, float * layer_out);  // forward of one token; logits into logits_
    // forward() of the current columns (cols_, tok_p_, pos_p_), through graph `exec` if `graph` and use_graph_.
    void run_forward(bool graph, cudaGraphExec_t & exec);
    void alloc_prefill();
    void prefill_chunk(const int32_t * ids_host, int n, int p0, float * logits);
    void attention_block(const DeviceLayer & L, bool swa);
    void forward();
    void ffn_block(const DeviceLayer & L, bool dense);
    void gemv(const DeviceMat & w, const float * x, float * y, int64_t ncols);  // cols_ columns
    void trace(const char * name, const float * d);
    int cur_layer_ = -1;
    // Columns of the current forward: 1 (step) or W (verify); tokens and position of the first column in GPU memory.
    int cols_ = 1;
    const int32_t * tok_p_ = nullptr;
    const int32_t * pos_p_ = nullptr;

    const Model & m_;
    EngineOptions opt_;
    cudaStream_t st_ = nullptr;
    std::vector<DeviceLayer> layers_;
    DeviceMat output_norm_, output_, tok_embd_;
    std::vector<void *> allocs_;
    size_t weight_bytes_ = 0;

    // Activation buffers (F32 unless noted), for max(1, verify_cols) columns.
    float *x_ = nullptr, *h_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr, *o_ = nullptr;
    float *router_ = nullptr, *wsel_ = nullptr, *gate_ = nullptr, *up_ = nullptr, *act_ = nullptr, *escale_ = nullptr;
    float *dexp_ = nullptr, *sh_ = nullptr, *ffn_ = nullptr, *scores_ = nullptr, *logits_ = nullptr;
    int32_t * ids_ = nullptr;
    void * xq_ = nullptr;
    int32_t * io_ = nullptr;       // GPU: [token, pos] of the current token
    int32_t * io_host_ = nullptr;  // pinned host copy, copied into io_ before each token
    int32_t * vio_ = nullptr;       // GPU: [p0, ids[0..8), argmax[0..8)] of verify()
    int32_t * vio_host_ = nullptr;  // pinned host
    cudaGraphExec_t verify_exec_[9] = {};  // one graph per W
    float * layer_buf_ = nullptr;  // host destination of layer outputs (diagnostics), or null
    uint32_t * route_counts_ = nullptr;  // GPU, [n_layer][n_expert], if count_routing
    ExpertPlacement placement_;
    std::vector<void *> host_allocs_;    // pinned copies of non-resident experts
    // Prefill buffers, for at most opt_.ubatch tokens (alloc_prefill).
    struct Prefill {
        float *x = nullptr, *h = nullptr, *qkv = nullptr, *q = nullptr, *att = nullptr, *o = nullptr;
        float *router = nullptr, *wsel = nullptr, *escale = nullptr, *act = nullptr, *up = nullptr, *gate = nullptr;
        float *dexp = nullptr, *sh = nullptr, *logits = nullptr;
        int32_t * ids = nullptr, * tok = nullptr;
        void * xq = nullptr, * mask = nullptr, * xb = nullptr;  // xb: BF16 router input (cuBLAS)
        MmqIdsScratch mm;
        int mask_kv = 0;  // mask columns (n_ctx rounded up to 256)
        void * kf16 = nullptr, * vf16 = nullptr;  // K, V of global layers dequantized to F16 (kv_global_q8), mask_kv rows
    } pf_;
    ::cublasContext * cublas_ = nullptr;  // prefill router (cublasHandle_t)
    bool use_graph_ = true;
    cudaGraphExec_t graph_exec_ = nullptr;
    uint32_t * mask_dev_ = nullptr, * mask_pinned_ = nullptr;  // grammar mask (pick)
    uint32_t * vmask_dev_ = nullptr, * vmask_pinned_ = nullptr;  // verify masks, [verify_cols][words]
    cudaEvent_t pick_ev_ = nullptr;  // end of the copy of the chosen token to the host (pick_async)
    float * seen_ = nullptr;  // GPU, [n_vocab]: tokens chosen since reset_presence() (presence penalty)
};

}  // namespace fns
