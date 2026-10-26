// tests/mmq_parity.cu - prefill MMQ (fns::mmq, fns::mmq_experts) on real GGUF blocks.
//
//   fns-mmq-parity --model F.gguf [--tokens 77]
//
// For each MoE layer and one dense tensor per format:
//   1. expert pointer tables (half of the experts in VRAM, half in mapped host memory) against ggml's MMQ on the
//      contiguous tensor (same table, pointers base + e * size): bit-identical;
//   2. each MMQ output against the MMVQ GEMV of the matching token: per-output-row relative L2 error
//      (‖a - b‖ / ‖b‖), p99 below 2e-2. MMQ quantizes activations in blocks of 128 with half-precision partial
//      sums (K-quants) and sums in a different order than MMVQ: both differ by rounding only; an indexing
//      error gives an order-1 error.
// Exit code 1 at the first failure.
#include "fns/ggml_kernels.h"
#include "fns/model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { std::fprintf(stderr, "%s : %s\n", #x, cudaGetErrorString(err_)); std::exit(1); } } while (0)

template <class T> static T * dalloc(size_t n) {
    T * p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T) + 4096));
    CK(cudaMemset(p, 0, n * sizeof(T) + 4096));
    return p;
}

static std::vector<float> down(const float * d, size_t n) {
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// p99 over rows of the relative L2 error ‖a - b‖ / ‖b‖ (the worst case is printed separately: a row whose output
// nearly cancels has a large relative error without indicating a fault).
static double rel_err(const std::vector<float> & a, const std::vector<float> & b, size_t rows, size_t n,
                      double * worst_out = nullptr) {
    std::vector<double> e(rows);
    for (size_t r = 0; r < rows; ++r) {
        double num = 0, den = 1e-30;
        for (size_t i = 0; i < n; ++i) {
            const double d = (double) a[r * n + i] - b[r * n + i];
            num += d * d;
            den += (double) b[r * n + i] * b[r * n + i];
        }
        e[r] = std::sqrt(num / den);
    }
    std::sort(e.begin(), e.end());
    if (worst_out) *worst_out = e.back();
    return e[std::min(rows - 1, (size_t) (0.99 * rows))];
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int T = 77;  // token count not a multiple of the tile size, to exercise the edges
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--model")) model_path = argv[i + 1];
        else if (!std::strcmp(argv[i], "--tokens")) T = std::atoi(argv[i + 1]);
    }
    if (!model_path) {
        std::fprintf(stderr, "usage: fns-mmq-parity --model F.gguf [--tokens 77]\n");
        return 2;
    }
    fns::Model m(model_path);
    const int E = m.hp.n_embd, F = m.hp.n_ff_exp, K = m.hp.n_expert_used, NE = m.hp.n_expert;
    const int NMAX = 6144;
    float * x = dalloc<float>((size_t) T * K * NMAX);
    float * ya = dalloc<float>((size_t) T * K * NMAX);
    float * yb = dalloc<float>((size_t) T * K * NMAX);
    void * xq = dalloc<char>(fns::mmq_q8_1_bytes(NMAX, (size_t) T * K));
    void * xq1 = dalloc<char>(1 << 20);
    int32_t * ids = dalloc<int32_t>((size_t) T * K);
    fns::MmqIdsScratch s;
    s.ids_src1 = dalloc<int32_t>((size_t) T * K + 128);
    s.ids_dst = dalloc<int32_t>((size_t) T * K + 128);
    s.expert_bounds = dalloc<int32_t>(NE + 1);
    s.xq = dalloc<char>(fns::mmq_q8_1_bytes(NMAX, (size_t) T * K));
    const char ** table = dalloc<const char *>(NE);
    const char ** table_c = dalloc<const char *>(NE);

    std::vector<float> hx((size_t) T * K * NMAX);
    uint32_t r = 12345;
    for (float & v : hx) {
        r = r * 1664525u + 1013904223u;
        v = ((int) (r >> 9) - (1 << 22)) / (float) (1 << 22);
    }
    CK(cudaMemcpy(x, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
    // Synthetic routing: K distinct experts per token, spread over the NE experts (some receive no tokens).
    std::vector<int32_t> hid((size_t) T * K);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < K; ++j) hid[(size_t) t * K + j] = (t * 37 + j * 41 + (t * j) % 7) % NE;
    for (int t = 0; t < T; ++t) {  // experts are distinct within each token
        for (int j = 1; j < K; ++j)
            for (int i = 0; i < j; ++i)
                if (hid[(size_t) t * K + j] == hid[(size_t) t * K + i]) hid[(size_t) t * K + j] = (hid[(size_t) t * K + j] + 1) % NE, i = -1;
    }
    CK(cudaMemcpy(ids, hid.data(), hid.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    int checked = 0;
    const double tol = 2e-2;
    auto check_vs_mmvq = [&](const char * what, ggml_type type, const char * const * htab, int64_t ncols, int64_t nrows,
                             int64_t x_rows, const std::vector<float> & got) {
        // Reference: MMVQ of token t, expert ids[t][j], input row t (gate/up) or t*K+j (down).
        std::vector<float> ref((size_t) T * K * nrows);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < K; ++j) {
                const float * xr = x + ((size_t) t * x_rows + (x_rows == 1 ? 0 : j)) * ncols;
                fns::quantize_q8_1(xr, xq1, ncols, 1, nullptr);
                fns::mmvq(type, htab[hid[(size_t) t * K + j]], ncols, nrows, xq1, yb, 1, nullptr);
                CK(cudaMemcpy(ref.data() + ((size_t) t * K + j) * nrows, yb, nrows * sizeof(float), cudaMemcpyDeviceToHost));
            }
        double worst = 0;
        const double e = rel_err(got, ref, (size_t) T * K, nrows, &worst);
        if (!(e < tol)) {
            std::fprintf(stderr, "%s : p99 relative error %.3g against MMVQ (threshold %.0e)\n", what, e, tol);
            std::exit(1);
        }
        return std::pair{e, worst};
    };

    for (size_t il = 0; il < m.layers.size(); ++il) {
        const fns::Layer & L = m.layers[il];
        if (L.dense) continue;
        for (const fns::Tensor * tw : {&L.gate_exps, &L.down_exps}) {
            const int64_t ncols = tw->ne[0], nrows = tw->ne[1];
            const size_t eb = tw->nbytes / NE;
            const int64_t x_rows = tw == &L.gate_exps ? 1 : K;
            char * full = dalloc<char>(tw->nbytes);
            CK(cudaMemcpy(full, tw->data, tw->nbytes, cudaMemcpyHostToDevice));
            // Mixed table: slots of off + eb bytes (the matrix at +off, like up or down in [gate | up | down]).
            const size_t off = 4096, sl = off + eb;
            char * dev = dalloc<char>(sl * NE / 2);
            char * host = nullptr;
            CK(cudaHostAlloc(&host, sl * NE / 2 + 4096, cudaHostAllocMapped));
            std::memset(host, 0, sl * NE / 2 + 4096);
            std::vector<const char *> ht(NE), hc(NE);
            for (int e = 0; e < NE; ++e) {
                hc[e] = full + e * eb;
                if (e % 2 == 0) {
                    CK(cudaMemcpy(dev + (e / 2) * sl + off, (const char *) tw->data + e * eb, eb, cudaMemcpyHostToDevice));
                    ht[e] = dev + (e / 2) * sl;
                } else {
                    std::memcpy(host + (e / 2) * sl + off, (const char *) tw->data + e * eb, eb);
                    void * dp;
                    CK(cudaHostGetDevicePointer(&dp, host + (e / 2) * sl, 0));
                    ht[e] = (const char *) dp;
                }
            }
            CK(cudaMemcpy(table, ht.data(), NE * sizeof(char *), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(table_c, hc.data(), NE * sizeof(char *), cudaMemcpyHostToDevice));
            fns::mmq_experts(tw->type, table_c, 0, ncols, nrows, NE, x, x_rows, T, ids, K, s, ya, nullptr);
            fns::mmq_experts(tw->type, table, off, ncols, nrows, NE, x, x_rows, T, ids, K, s, yb, nullptr);
            CK(cudaDeviceSynchronize());
            const std::vector<float> a = down(ya, (size_t) T * K * nrows), b = down(yb, (size_t) T * K * nrows);
            if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
                std::fprintf(stderr, "layer %zu %s : mixed VRAM/host table differs from the contiguous table\n", il,
                             tw->name.c_str());
                return 1;
            }
            const auto [e, worst] = check_vs_mmvq(tw->name.c_str(), tw->type, hc.data(), ncols, nrows, x_rows, a);
            std::printf("layer %zu %s %s : table = contiguous bit-exact; relative error vs MMVQ p99 %.2e, max %.2e\n", il,
                        tw->name.c_str(), ggml_type_name(tw->type), e, worst);
            ++checked;
            CK(cudaFreeHost(host));
            CK(cudaFree(dev));
            CK(cudaFree(full));
        }
    }
    // Denses : un tenseur par format.
    std::vector<const fns::Tensor *> dense;
    auto take = [&](const fns::Tensor & t) {
        if (!t.data) return;
        for (const fns::Tensor * d : dense)
            if (d->type == t.type) return;
        dense.push_back(&t);
    };
    take(m.output);
    for (const fns::Layer & L : m.layers)
        for (const fns::Tensor * t : {&L.wq, &L.wo, &L.ffn_down, &L.ffn_gate, &L.gate_shexp, &L.down_shexp}) take(*t);
    for (const fns::Tensor * t : dense) {
        const int64_t ncols = t->ne[0], nrows = t->ne[1];
        char * w = dalloc<char>(t->nbytes);
        CK(cudaMemcpy(w, t->data, t->nbytes, cudaMemcpyHostToDevice));
        float * y = dalloc<float>((size_t) T * nrows);
        fns::mmq(t->type, w, ncols, nrows, x, T, xq, y, nullptr);
        CK(cudaDeviceSynchronize());
        const std::vector<float> got = down(y, (size_t) T * nrows);
        std::vector<float> ref((size_t) T * nrows);
        float * yr = dalloc<float>(nrows);
        for (int tk = 0; tk < T; ++tk) {
            fns::quantize_q8_1(x + (size_t) tk * ncols, xq1, ncols, 1, nullptr);
            fns::mmvq(t->type, w, ncols, nrows, xq1, yr, 1, nullptr);
            CK(cudaMemcpy(ref.data() + (size_t) tk * nrows, yr, nrows * sizeof(float), cudaMemcpyDeviceToHost));
        }
        double worst = 0;
        const double e = rel_err(got, ref, T, nrows, &worst);
        std::printf("dense %s %s [%lld x %lld] : relative error vs MMVQ p99 %.2e, max %.2e\n", t->name.c_str(),
                    ggml_type_name(t->type), (long long) nrows, (long long) ncols, e, worst);
        if (!(e < tol)) return 1;
        ++checked;
        CK(cudaFree(w));
        CK(cudaFree(y));
        CK(cudaFree(yr));
    }
    std::printf("%d checks passed\n", checked);
    return 0;
}
