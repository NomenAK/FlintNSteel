// tests/mmvq_bench.cu - timing of expert GEMVs on real GGUF blocks, in isolation (diagnostic).
//
//   fns-mmvq-bench --model F.gguf [--layer 3] [--reps 200]
//
// For the given layer: gate+up+SwiGLU experts (mmvq_ids_swiglu_shared, 6 experts, 1 input), down (mmvq_ids, 6
// inputs), and for comparison a dense GEMV over the same bytes. Median time (CUDA events) and effective GB/s.
#include "fns/ggml_kernels.h"
#include "fns/model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } } while (0)

template <class F> static float median_us(int reps, F && f) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    f();
    CK(cudaDeviceSynchronize());
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a));
        f();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms * 1000.0f);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static void * upload(const fns::Tensor & t) {
    void * d = nullptr;
    CK(cudaMalloc(&d, t.nbytes + 4096));
    CK(cudaMemset(d, 0, t.nbytes + 4096));
    CK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int layer = 3, reps = 200;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--model")) model_path = argv[i + 1];
        else if (!std::strcmp(argv[i], "--layer")) layer = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[i + 1]);
    }
    if (!model_path) {
        std::fprintf(stderr, "usage: fns-mmvq-bench --model F.gguf [--layer N] [--reps N]\n");
        return 2;
    }
    fns::Model m(model_path);
    const fns::Layer & L = m.layers[(size_t) layer];
    const int E = m.hp.n_embd, F = m.hp.n_ff_exp, K = m.hp.n_expert_used, NE = m.hp.n_expert;
    void * g = upload(L.gate_exps), * u = upload(L.up_exps), * dn = upload(L.down_exps), * wq = upload(L.wq);
    float * x = nullptr, * y = nullptr;
    void * xq = nullptr;
    int32_t * ids = nullptr;
    CK(cudaMalloc(&x, 6 * 6144 * sizeof(float)));
    CK(cudaMalloc(&y, 6 * 6144 * sizeof(float)));
    CK(cudaMalloc(&xq, 1 << 20));
    CK(cudaMalloc(&ids, 8 * sizeof(int32_t)));
    std::vector<float> hx(6 * 6144, 0.25f);
    CK(cudaMemcpy(x, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
    const int32_t hid[6] = {3, 77, 140, 201, 9, 250};
    CK(cudaMemcpy(ids, hid, sizeof(hid), cudaMemcpyHostToDevice));
    fns::quantize_q8_1(x, xq, E, 1, nullptr);

    const double gu_bytes = 2.0 * K * (double) L.gate_exps.nbytes / NE;
    const double dn_bytes = (double) K * L.down_exps.nbytes / NE;
    const float t_gu = median_us(reps, [&] {
        fns::mmvq_ids_swiglu_shared(L.gate_exps.type, g, u, E, F, NE, xq, ids, K, y, nullptr, nullptr, nullptr, nullptr);
    });
    fns::quantize_q8_1(x, xq, F, K, nullptr);
    const float t_dn = median_us(reps, [&] { fns::mmvq_ids(L.down_exps.type, dn, F, E, NE, xq, K, ids, K, y, nullptr); });
    fns::quantize_q8_1(x, xq, E, 1, nullptr);
    const float t_q = median_us(reps, [&] { fns::mmvq(L.wq.type, wq, E, L.wq.ne[1], xq, y, 1, nullptr); });
    std::printf("layer %d  gate+up %s  %.1f us  %.0f GB/s\n", layer, ggml_type_name(L.gate_exps.type), t_gu,
                gu_bytes / t_gu / 1e3);
    std::printf("layer %d  down    %s  %.1f us  %.0f GB/s\n", layer, ggml_type_name(L.down_exps.type), t_dn,
                dn_bytes / t_dn / 1e3);
    std::printf("layer %d  attn_q  %s  %.1f us  %.0f GB/s\n", layer, ggml_type_name(L.wq.type), t_q,
                (double) L.wq.nbytes / t_q / 1e3);
    return 0;
}
