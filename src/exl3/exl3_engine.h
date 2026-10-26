// Flint EXL3 engine: QTIP weights in VRAM, zero-copy BF16 embedding, F16 ring-buffer KV cache.
#pragma once
#include "fns/engine.h"
#include "exl3_model.h"
#include "exl3_kernels.h"
#include <cuda_bf16.h>
namespace fns {
class Exl3Engine {
  public:
    Exl3Engine(const Exl3Model &model, const EngineOptions &opt);
    ~Exl3Engine();
    Exl3Engine(const Exl3Engine &) = delete;
    Exl3Engine &operator=(const Exl3Engine &) = delete;
    void step(int32_t token, int pos, float *logits, float *layer_out = nullptr);
    int32_t step_greedy(int32_t token, int pos);
    void step_logits(int32_t token, int pos);
    int32_t pick(const uint32_t *mask, const SampleParams &sp = {}, uint64_t counter = 0);
    void pick_async(const uint32_t *mask, const SampleParams &sp = {}, uint64_t counter = 0);
    int32_t pick_wait();
    void step_next(int pos);
    void reset_presence();
    int32_t prefill(const int32_t *ids, int n, int p0, float *logits = nullptr);
    size_t weight_bytes() const { return weight_bytes_; }
    const ExpertPlacement &placement() const { return placement_; }
    int ring() const { return opt_.ring; }

    friend class Exl3Diagnostic;
  private:
    struct Layer {
        float *an = nullptr, *fn = nullptr, *qn = nullptr, *kn = nullptr, *sinks = nullptr;
        exl3::Router router;
        exl3::Linear q, k, v, o, g, u, d, sg, su, sd;
        MoeCoopParams moe = {};
        MoeCoopParams shared = {};
        void *kc = nullptr, *vc = nullptr;
        int slots = 0;
    };
    void *allocate(size_t bytes);
    void *upload(const Exl3Tensor &t);
    float *norm(const std::string &name);
    exl3::Linear linear(const std::string &key);
    void project(const exl3::Linear &w, const float *x, float *y, int rows, bool half_output = false);
    // Only the test friend instantiates Trace=true; normal forwards compile out every trace point.
    struct Trace {
        void *context;
        void (*visit)(void *, int, int, const char *, void *, int, int, bool, cudaStream_t);
    };
    template <bool TraceEnabled = false>
    void forward(int rows, int p0, bool head_all, float *host_logits = nullptr, float *layers = nullptr,
                 Trace *trace = nullptr);
    static void decode_attention(const float *q, const half *k, const half *v, float *part,
                                 const float *sinks, float *out, const int32_t *position,
                                 int slots, int window, int chunks, cudaStream_t stream);
    void run_decode(float *layers = nullptr);
    void launch(int32_t token, int pos, float *layers = nullptr);
    const Exl3Model &m_;
    EngineOptions opt_;
    cudaStream_t st_ = nullptr;
    cudaGraphExec_t graph_ = nullptr;
    cudaEvent_t pick_ev_ = nullptr;
    ::cublasContext *blas_ = nullptr;
    std::vector<Layer> layers_;
    std::vector<void *> allocs_;
    void *embedding_host_ = nullptr;
    const __nv_bfloat16 *embedding_ = nullptr;
    float *norm_ = nullptr;
    exl3::Linear head_;
    size_t weight_bytes_ = 0;
    ExpertPlacement placement_;
    float *x_ = nullptr, *h_ = nullptr, *qkv_ = nullptr, *q_ = nullptr, *att_ = nullptr, *out_ = nullptr;
    float *gate_ = nullptr, *up_ = nullptr, *act_ = nullptr, *shared_ = nullptr, *routed_ = nullptr, *router_ = nullptr,
          *rw_f_ = nullptr;
    float *scores_ = nullptr, *logits_ = nullptr, *seen_ = nullptr;
    half *xh_ = nullptr, *had_ = nullptr, *yh_ = nullptr, *rw_ = nullptr, *mask_ = nullptr;
    int32_t *io_ = nullptr, *host_io_ = nullptr, *tokens_ = nullptr, *ids_ = nullptr;
    int64_t *selected_ = nullptr;
    int *locks_ = nullptr;
    uint32_t *mask_dev_ = nullptr, *mask_host_ = nullptr;
    MoeCoopParams scratch_ = {};
    MoeCoopParams shared_scratch_ = {};
    int64_t *shared_sel_ = nullptr;
    half *shared_rw_ = nullptr;
    exl3::RouterScratch router_scratch_;
    int mask_cols_ = 0;
};
} // namespace fns
