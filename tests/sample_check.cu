// tests/sample_check.cu - k::sample frequency check against the host-computed top-k / top-p / temperature distribution,
// on synthetic logits, with and without a mask.
//
//   fns-sample-check
//
// 200 000 draws (counters 0 .. N-1) per case. Criteria: |frequency - probability| < 5 binomial standard deviations for
// each candidate, no draw outside the top-k / top-p set or outside the mask, temperature 0 = masked argmax, and exact
// argmax (smallest index) for ties spread across several top-k slices in two passes.
#include "../src/kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(err_)); std::exit(1); } } while (0)

static int run_case(const char * name, const std::vector<float> & logits, const std::vector<uint32_t> * mask, int top_k,
                    float top_p, float temp) {
    const int n = (int) logits.size(), N = 200000;
    float * dx;
    uint32_t * dm = nullptr;
    int32_t * dout;
    CK(cudaMalloc(&dx, n * sizeof(float)));
    CK(cudaMalloc(&dout, N * sizeof(int32_t)));
    CK(cudaMemcpy(dx, logits.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    if (mask) {
        CK(cudaMalloc(&dm, mask->size() * sizeof(uint32_t)));
        CK(cudaMemcpy(dm, mask->data(), mask->size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    for (int i = 0; i < N; ++i) fns::k::sample(dx, n, dm, top_k, top_p, temp, 1234, (uint64_t) i, dout + i, nullptr);
    std::vector<int32_t> out(N);
    CK(cudaMemcpy(out.data(), dout, N * sizeof(int32_t), cudaMemcpyDeviceToHost));
    // Host reference: allowed candidates sorted (logit descending, index ascending), top-k, top-p on softmax T=1,
    // then temperature softmax over the kept candidates.
    std::vector<int> idx;
    for (int i = 0; i < n; ++i)
        if (!mask || (((*mask)[i >> 5] >> (i & 31)) & 1u)) idx.push_back(i);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    const int k = std::min<int>((int) idx.size(), top_k <= 0 || top_k > 64 ? 64 : top_k);
    idx.resize(k);
    std::vector<double> p(k);
    double s = 0;
    for (int j = 0; j < k; ++j) s += p[j] = std::exp((double) logits[idx[j]] - logits[idx[0]]);
    int keep = k;
    double cum = 0;
    for (int j = 0; j < k; ++j) {
        cum += p[j] / s;
        if (cum >= top_p) { keep = j + 1; break; }
    }
    std::vector<double> w(keep);
    double ws = 0;
    for (int j = 0; j < keep; ++j) ws += w[j] = std::exp(((double) logits[idx[j]] - logits[idx[0]]) / temp);
    std::vector<int> count(n, 0);
    for (int t : out) {
        if (t < 0 || t >= n) { std::fprintf(stderr, "%s : draw %d out of vocabulary\n", name, t); return 1; }
        ++count[t];
    }
    int bad = 0, outside = N;
    double worst = 0;
    for (int j = 0; j < keep; ++j) {
        const double q = w[j] / ws, f = (double) count[idx[j]] / N, sd = std::sqrt(q * (1 - q) / N) + 1e-12;
        worst = std::max(worst, std::fabs(f - q) / sd);
        if (std::fabs(f - q) > 5 * sd) ++bad;
        outside -= count[idx[j]];
    }
    std::printf("%-28s kept %2d, max deviation %.2f sigma, outside %d\n", name, keep, worst, outside);
    CK(cudaFree(dx));
    CK(cudaFree(dout));
    if (dm) CK(cudaFree(dm));
    return bad || outside ? 1 : 0;
}

int main() {
    const int n = 150144;
    std::vector<float> logits(n);
    uint32_t r = 7;
    for (float & v : logits) {
        r = r * 1664525u + 1013904223u;
        v = ((int) (r >> 9) - (1 << 22)) / (float) (1 << 22) * 3.0f;
    }
    for (int j = 0; j < 30; ++j) logits[(j * 4999) % n] = 8.0f - 0.25f * j;  // a sharp peak of 30 tokens
    std::vector<uint32_t> mask((n + 31) / 32, 0);
    for (int j = 0; j < 30; j += 2) mask[((j * 4999) % n) >> 5] |= 1u << (((j * 4999) % n) & 31);  // every other one
    int fail = 0;
    fail |= run_case("top-k 20, top-p 0.95, T 1", logits, nullptr, 20, 0.95f, 1.0f);
    fail |= run_case("top-k 64, top-p 1, T 0.7", logits, nullptr, 0, 1.0f, 0.7f);
    fail |= run_case("masked, top-k 20, T 1", logits, &mask, 20, 1.0f, 1.0f);
    // temperature 0: masked argmax.
    float * dx;
    uint32_t * dm;
    int32_t * dout, got;
    CK(cudaMalloc(&dx, n * sizeof(float)));
    CK(cudaMalloc(&dm, mask.size() * sizeof(uint32_t)));
    CK(cudaMalloc(&dout, sizeof(int32_t)));
    CK(cudaMemcpy(dx, logits.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dm, mask.data(), mask.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    fns::k::sample(dx, n, dm, 20, 0.95f, 0.0f, 0, 0, dout, nullptr);
    CK(cudaMemcpy(&got, dout, sizeof(int32_t), cudaMemcpyDeviceToHost));
    const bool argmax_ok = got == 0;  // logits[0] = 8 (j = 0), allowed
    std::printf("T 0 masked : %d (%s)\n", got, argmax_ok ? "argmax" : "WRONG");
    // Exact two-pass top-k (k_topk_part then k_sample): logit ties spread across several slices, with a mask.
    // At T 0 the result must be the argmax of the total order (logit descending, index ascending).
    int tie_bad = 0;
    std::vector<float> flat(n, 1.0f);
    for (int i : {149000, 70000, 1201, 1200}) flat[i] = 5.0f;  // tie at 5 in four slices: 1200 expected
    CK(cudaMemcpy(dx, flat.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    fns::k::argmax(dx, n, dout, nullptr);
    CK(cudaMemcpy(&got, dout, sizeof(int32_t), cudaMemcpyDeviceToHost));
    tie_bad |= got != 1200;
    std::vector<uint32_t> m2((n + 31) / 32, 0);
    m2[70000 >> 5] |= 1u << (70000 & 31);
    m2[149000 >> 5] |= 1u << (149000 & 31);
    CK(cudaMemcpy(dm, m2.data(), m2.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    fns::k::argmax(dx, n, dout, nullptr, dm);
    CK(cudaMemcpy(&got, dout, sizeof(int32_t), cudaMemcpyDeviceToHost));
    tie_bad |= got != 70000;
    std::fill(flat.begin(), flat.end(), 2.0f);  // all equal: 0
    CK(cudaMemcpy(dx, flat.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    fns::k::argmax(dx, n, dout, nullptr);
    CK(cudaMemcpy(&got, dout, sizeof(int32_t), cudaMemcpyDeviceToHost));
    tie_bad |= got != 0;
    std::fill(m2.begin(), m2.end(), 0u);  // empty mask: -1
    CK(cudaMemcpy(dm, m2.data(), m2.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    fns::k::sample(dx, n, dm, 20, 1.0f, 1.0f, 0, 0, dout, nullptr);
    CK(cudaMemcpy(&got, dout, sizeof(int32_t), cudaMemcpyDeviceToHost));
    tie_bad |= got != -1;
    std::printf("ties, mask, empty mask : %s\n", tie_bad ? "WRONG" : "ok");
    return fail || !argmax_ok || tie_bad ? 1 : 0;
}
