// tests/attention_mma_parity.cu - global MMA attention against ggml on Q/K/V from real GGUF weights.
// Real tokens feed embedding -> RMSNorm -> QKV -> Q/K norm (first global layer).
// Checks precision against ggml's FA and internal consistency of W1..W8, including split-K boundaries.
#include "fns/model.h"
#include "fns/ggml_kernels.h"
#include "../src/kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)
int main(int argc, char ** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: fns-attention-mma-parity model.gguf tokens.txt\n"); return 2; }
    fns::Model m(argv[1]);
    std::ifstream input(argv[2]); std::stringstream text; text << input.rdbuf();
    std::string raw = text.str(); std::replace(raw.begin(), raw.end(), ',', ' '); std::istringstream words(raw);
    std::vector<int32_t> tokens; int32_t token; while (words >> token) tokens.push_back(token);
    if (tokens.size() < 1024) return 2;
    constexpr int N = 1024, E = 2048, QD = 4096, KVD = 512, STRIDE = QD + 2 * KVD;
    // A null pointer makes ggml create a private non-blocking stream. The explicit legacy handle
    // keeps the synchronous copy order of this test without changing the kernels.
    const cudaStream_t st = cudaStreamLegacy;
    const fns::Layer * L = nullptr;
    for (const auto & l : m.layers) if (!l.swa) { L = &l; break; }
    if (!L) return 2;
    std::vector<void *> allocs;
    auto alloc = [&](size_t n) { void * p = nullptr; if (cudaMalloc(&p, n) != cudaSuccess) std::abort(); allocs.push_back(p); cudaMemset(p, 0, n); return p; };
    auto upload = [&](const fns::Tensor & t) { void * p = alloc(t.nbytes + 4096); cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice); return p; };
    void * emb = upload(m.tok_embd), * norm = upload(L->attn_norm), * qn = upload(L->q_norm), * kn = upload(L->k_norm), * sinks = upload(L->sinks);
    void * qw = upload(L->wq), * kw = upload(L->wk), * vw = upload(L->wv);
    auto * ids = (int32_t *) alloc(N * sizeof(int32_t));
    CK(cudaMemcpy(ids, tokens.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
    auto * x = (float *) alloc(N * E * sizeof(float)); auto * h = (float *) alloc(N * E * sizeof(float));
    auto * qkv = (float *) alloc(N * STRIDE * sizeof(float));
    auto * q = (float *) alloc(N * QD * sizeof(float));
    auto * proj = (float *) alloc(N * QD * sizeof(float));
    void * xq = alloc(fns::mmq_q8_1_bytes(E, N));
    void * K = alloc(N * KVD * sizeof(__half)), * V = alloc(N * KVD * sizeof(__half));
    fns::k::embed_q8_0_batch(emb, ids, x, E, N, st);
    fns::k::rms_norm_mul(x, (const float *) norm, h, E, N, m.hp.rms_eps, st);
    for (int part = 0; part < 3; ++part) {
        const fns::Tensor & t = part == 0 ? L->wq : part == 1 ? L->wk : L->wv;
        const void * w = part == 0 ? qw : part == 1 ? kw : vw;
        const int width = part == 0 ? QD : KVD, offset = part == 0 ? 0 : part == 1 ? QD : QD + KVD;
        fns::mmq(t.type, w, E, width, h, N, xq, proj, st);
        CK(cudaMemcpy2D(qkv + offset, STRIDE * sizeof(float), proj, width * sizeof(float), width * sizeof(float), N, cudaMemcpyDeviceToDevice));
    }
    fns::k::qk_norm_rope_store_batch(qkv, STRIDE, q, (const float *) qn, (const float *) kn, K, V, 0, N, 32, 4, 128, N,
                                   m.hp.rms_eps, m.hp.rope_base, false, st);
    auto * pos = (int32_t *) alloc(sizeof(int32_t));
    const size_t ps = fns::k::attention_part_floats(32, 128, N);
    auto * scratch = (float *) alloc(8 * ps * sizeof(float));
    auto * result = (float *) alloc(8 * QD * sizeof(float)); auto * reference = (float *) alloc(8 * QD * sizeof(float));
    void * mask = alloc(8 * N * sizeof(__half));
    std::vector<float> a(8 * QD), b(8 * QD), scalar(8 * QD);
    int failed = 0, exact = 0, cases = 0;
    for (int p : {0, 15, 16, 62, 63, 64, 510, 511, 512, 1000}) {
        for (int W : {1, 2, 3, 4, 5, 6, 7, 8}) {
            CK(cudaMemcpy(pos, &p, sizeof(p), cudaMemcpyHostToDevice));
            fns::k::attention(q + (size_t) p * QD, K, V, (const float *) sinks, result, scratch, 32, 4, 128, pos, 0, N, N,
                              1.0f / std::sqrt(128.0f), W, QD, st, false, true);
            CK(cudaMemcpy(a.data(), result, W * QD * sizeof(float), cudaMemcpyDeviceToHost));
            for (int c = 0; c < W; ++c) {
                const int pp = p + c; CK(cudaMemcpy(pos, &pp, sizeof(pp), cudaMemcpyHostToDevice));
                fns::k::attention(q + (size_t) pp * QD, K, V, (const float *) sinks, result, scratch, 32, 4, 128, pos, 0, N, N,
                                  1.0f / std::sqrt(128.0f), 1, QD, st, false, true);
                CK(cudaMemcpy(scalar.data() + c * QD, result, QD * sizeof(float), cudaMemcpyDeviceToHost));
            }
            const bool same = std::memcmp(a.data(), scalar.data(), W * QD * sizeof(float)) == 0;
            exact += same;
            fns::k::prefill_mask(mask, N, p, W, N, 0, st);
            fns::flash_attn_prefill(q + (size_t) p * QD, K, V, mask, (const float *) sinks, reference, W, N, 32, 4, 128,
                                    1.0f / std::sqrt(128.0f), st);
            CK(cudaMemcpy(b.data(), reference, W * QD * sizeof(float), cudaMemcpyDeviceToHost));
            double err = 0, denom = 0, maxerr = 0;
            for (int i = 0; i < W * QD; ++i) { const double d = a[i] - b[i]; err += d*d; denom += double(b[i])*b[i]; maxerr = std::max(maxerr, std::abs(d)); }
            const double rel = std::sqrt(err / std::max(denom, 1e-30));
            const bool ok = same && std::isfinite(rel) && rel < 0.01;
            failed += !ok; ++cases;
            std::printf("p=%d W=%d seq_exact=%d rel_ggml=%.7g max=%.7g %s\n", p, W, same, rel, maxerr, ok ? "PASS" : "FAIL");
        }
    }
    std::printf("cases=%d exact=%d failed=%d\n", cases, exact, failed);
    for (void * p : allocs) cudaFree(p);
    return failed ? 1 : 0;
}
