// tests/expert_parity.cu - parity of expert GEMVs via pointer tables (mmvq_experts*) against ggml (mmvq_ids*).
//
//   fns-expert-parity --model F.gguf
//
// For each MoE layer: gate+up+SwiGLU (one input row) and down (six input rows), on real GGUF blocks.
// The table points half of the experts at a VRAM copy and the other half at pinned, mapped host memory
// (zero-copy). Pass criterion: results bit-identical to mmvq_ids_swiglu_shared / mmvq_ids on the full
// VRAM tensor. Exit code 1 at the first mismatch.
#include "fns/ggml_kernels.h"
#include "fns/model.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { std::fprintf(stderr, "%s : %s\n", #x, cudaGetErrorString(err_)); std::exit(1); } } while (0)

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i + 1 < argc; i += 2)
        if (!std::strcmp(argv[i], "--model")) model_path = argv[i + 1];
    if (!model_path) {
        std::fprintf(stderr, "usage: fns-expert-parity --model F.gguf\n");
        return 2;
    }
    fns::Model m(model_path);
    const int E = m.hp.n_embd, F = m.hp.n_ff_exp, K = m.hp.n_expert_used, NE = m.hp.n_expert;

    float * x = nullptr, * ya = nullptr, * yb = nullptr;
    void * xq = nullptr;
    int32_t * ids = nullptr;
    const char ** table = nullptr;
    CK(cudaMalloc(&x, (size_t) K * E * sizeof(float)));
    CK(cudaMalloc(&ya, (size_t) K * E * sizeof(float)));
    CK(cudaMalloc(&yb, (size_t) K * E * sizeof(float)));
    CK(cudaMalloc(&xq, 1 << 20));
    CK(cudaMalloc(&ids, K * sizeof(int32_t)));
    CK(cudaMalloc(&table, NE * sizeof(char *)));
    std::vector<float> hx((size_t) K * E);
    uint32_t s = 12345;
    for (float & v : hx) {
        s = s * 1664525u + 1013904223u;
        v = ((int) (s >> 9) - (1 << 22)) / (float) (1 << 22);
    }
    CK(cudaMemcpy(x, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> ha((size_t) K * E), hb((size_t) K * E);
    int checked = 0;
    for (size_t il = 0; il < m.layers.size(); ++il) {
        const fns::Layer & L = m.layers[il];
        if (L.dense) continue;
        // Per-expert slot: [gate | up | down], as in the engine's expert cache.
        const size_t gb = L.gate_exps.nbytes / NE, ub = L.up_exps.nbytes / NE, db = L.down_exps.nbytes / NE;
        const size_t slot = gb + ub + db, pad = 4096;
        void * dev_slots = nullptr, * host_slots = nullptr, * full_g = nullptr, * full_u = nullptr, * full_d = nullptr;
        CK(cudaMalloc(&dev_slots, NE / 2 * slot + pad));
        CK(cudaHostAlloc(&host_slots, NE / 2 * slot + pad, cudaHostAllocMapped));
        std::vector<const char *> ht(NE);
        for (int e = 0; e < NE; ++e) {
            char * dst;
            if (e % 2 == 0) {
                dst = (char *) dev_slots + (size_t) (e / 2) * slot;
                CK(cudaMemcpy(dst, (const char *) L.gate_exps.data + e * gb, gb, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dst + gb, (const char *) L.up_exps.data + e * ub, ub, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dst + gb + ub, (const char *) L.down_exps.data + e * db, db, cudaMemcpyHostToDevice));
                ht[e] = dst;
            } else {
                char * h = (char *) host_slots + (size_t) (e / 2) * slot;
                std::memcpy(h, (const char *) L.gate_exps.data + e * gb, gb);
                std::memcpy(h + gb, (const char *) L.up_exps.data + e * ub, ub);
                std::memcpy(h + gb + ub, (const char *) L.down_exps.data + e * db, db);
                void * d;
                CK(cudaHostGetDevicePointer(&d, h, 0));
                ht[e] = (const char *) d;
            }
        }
        CK(cudaMemcpy(table, ht.data(), NE * sizeof(char *), cudaMemcpyHostToDevice));
        for (auto [p, t] : {std::pair{&full_g, &L.gate_exps}, {&full_u, &L.up_exps}, {&full_d, &L.down_exps}}) {
            CK(cudaMalloc(p, t->nbytes + pad));
            CK(cudaMemcpy(*p, t->data, t->nbytes, cudaMemcpyHostToDevice));
        }
        // Even experts (VRAM) and odd experts (host) mixed, including the extremes 0 and 255.
        const int32_t hid[8] = {0, 255, 17, 128, 3, 200, 64, 9};
        CK(cudaMemcpy(ids, hid, K * sizeof(int32_t), cudaMemcpyHostToDevice));

        auto compare = [&](const char * what, size_t n) {
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(ha.data(), ya, n * sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hb.data(), yb, n * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(ha.data(), hb.data(), n * sizeof(float)) != 0) {
                size_t i = 0;
                while (std::memcmp(&ha[i], &hb[i], sizeof(float)) == 0) ++i;
                std::fprintf(stderr, "layer %zu %s : mismatch at index %zu (%.9g vs %.9g)\n", il, what, i, ha[i], hb[i]);
                std::exit(1);
            }
            ++checked;
        };
        fns::quantize_q8_1(x, xq, E, 1, nullptr);
        fns::mmvq_ids_swiglu_shared(L.gate_exps.type, full_g, full_u, E, F, NE, xq, ids, K, ya, nullptr, nullptr, nullptr,
                                    nullptr);
        fns::mmvq_experts_swiglu(L.gate_exps.type, table, 0, gb, E, F, xq, 1, ids, K, yb, nullptr);
        compare("gate+up", (size_t) K * F);
        fns::quantize_q8_1(x, xq, F, K, nullptr);
        fns::mmvq_ids(L.down_exps.type, full_d, F, E, NE, xq, K, ids, K, ya, nullptr);
        fns::mmvq_experts(L.down_exps.type, table, gb + ub, F, E, xq, K, ids, K, yb, nullptr);
        compare("down", (size_t) K * E);
        std::printf("layer %zu : gate+up %s, down %s identical\n", il, ggml_type_name(L.gate_exps.type),
                    ggml_type_name(L.down_exps.type));
        CK(cudaFree(dev_slots));
        CK(cudaFreeHost(host_slots));
        CK(cudaFree(full_g));
        CK(cudaFree(full_u));
        CK(cudaFree(full_d));
    }
    std::printf("%d comparisons, all bit-identical\n", checked);
    return 0;
}
