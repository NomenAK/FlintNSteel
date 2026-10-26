// tools/fns_verify/fns_verify.cpp - W-column verification checks (verify()): parity with the sequential path,
// causality, then full greedy speculative decoding (oracle or prompt-lookup draft).
//
//   fns-verify --model F.gguf [--prompt ids.txt | -d D] [-n 256] [-W 2,4,8] [-ub 2048] [--draft oracle|lookup|both]
//              [--parity-windows 8] [-o out.json]
//
// 1. Fill the context (the prompt, or D synthetic ids as fns-bench does), then generate n greedy tokens one step at a
//    time: reference sequence and time T1 per token (host clock, synchronized per token as in fns-bench).
// 2. Parity, for each W: over `parity-windows` windows of the reference sequence, sequential logits of the W positions
//    (step) against verify() on the same ids; max |Δ|, KL(seq || verify), identical argmax. The KV is then rewritten
//    by step. Causality: verify with a different suffix; columns before the change must stay bit-identical.
// 3. Decoding: from the state after the prompt, windows [last token, draft...] go to verify; column 0's choice is
//    committed, then choices continue while the draft equals the previous choice (exact greedy). Oracle draft: the
//    reference continuation, so every draft is accepted (upper bound on tokens per window). Lookup draft: the
//    continuation after the last occurrence of the final n-gram (n = 4 down to 2) in prompt + output. Without a draft,
//    one sequential step. Measures tok/s and tokens committed per window, and compares the produced sequence with the
//    reference.
#include "fns/engine.h"
#include "fns/lookup.h"
#include "fns/model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

std::vector<int32_t> read_ids(const char * path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream in(s);
    std::vector<int32_t> v;
    long x;
    while (in >> x) v.push_back((int32_t) x);
    return v;
}

using fns::lookup_draft;

struct Diff {
    double max_abs = 0, kl = 0;
    bool same_argmax = true;
};

Diff compare(const float * a, const float * b, int n) {
    Diff d;
    float ma = -INFINITY, mb = -INFINITY;
    int ia = 0, ib = 0;
    for (int i = 0; i < n; ++i) {
        d.max_abs = std::max(d.max_abs, (double) std::fabs(a[i] - b[i]));
        if (a[i] > ma) { ma = a[i]; ia = i; }
        if (b[i] > mb) { mb = b[i]; ib = i; }
    }
    double za = 0, zb = 0;
    for (int i = 0; i < n; ++i) {
        za += std::exp((double) a[i] - ma);
        zb += std::exp((double) b[i] - mb);
    }
    const double la = std::log(za), lb = std::log(zb);
    for (int i = 0; i < n; ++i) {
        const double pa = (double) a[i] - ma - la, pb = (double) b[i] - mb - lb;
        d.kl += std::exp(pa) * (pa - pb);
    }
    d.same_argmax = ia == ib;
    return d;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * model_path = nullptr, * prompt_path = nullptr, * out_path = nullptr;
    int depth = 0, n_gen = 256, ubatch = 2048, parity_windows = 8;
    std::vector<int> Ws = {2, 4, 8};
    std::string draft = "both";
    bool kv_q8 = false;
    bool attention_mma = false;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        const char * v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!std::strcmp(a, "--model") && v) { model_path = v; ++i; }
        else if (!std::strcmp(a, "--prompt") && v) { prompt_path = v; ++i; }
        else if (!std::strcmp(a, "-d") && v) { depth = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "-n") && v) { n_gen = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "-ub") && v) { ubatch = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--parity-windows") && v) { parity_windows = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--draft") && v) { draft = v; ++i; }
        else if (!std::strcmp(a, "-o") && v) { out_path = v; ++i; }
        else if (!std::strcmp(a, "--kv-global") && v) { kv_q8 = !std::strcmp(v, "q8_0"); ++i; }
        else if (!std::strcmp(a, "--attention-mma")) { attention_mma = true; }
        else if (!std::strcmp(a, "-W") && v) {
            Ws.clear();
            std::string s = v;
            std::replace(s.begin(), s.end(), ',', ' ');
            std::istringstream in(s);
            int w;
            while (in >> w) Ws.push_back(w);
            ++i;
        } else {
            std::fprintf(stderr, "usage: fns-verify --model F.gguf [--prompt ids.txt | -d D] [-n 256] [-W 2,4,8] "
                                 "[-ub 2048] [--draft oracle|lookup|both] [--parity-windows 8] [-o out.json] "
                                 "[--kv-global f16|q8_0] [--attention-mma]\n");
            return 2;
        }
    }
    if (!model_path || Ws.empty()) return 2;
    for (int w : Ws)
        if (w < 1 || w > 8) return 2;
    std::vector<int32_t> prompt;
    if (prompt_path) {
        prompt = read_ids(prompt_path);
        if (prompt.empty()) {
            std::fprintf(stderr, "fns-verify: no ids in %s\n", prompt_path);
            return 2;
        }
    } else {
        for (int p = 0; p < std::max(depth, 1); ++p) prompt.push_back((int32_t) (1000 + (p * 7919) % 100000));
    }
    const int P = (int) prompt.size();
    try {
        fns::Model model(model_path);
        fns::EngineOptions opt;
        opt.n_ctx = P + n_gen + 16;
        opt.ubatch = P > 1 ? ubatch : 0;
        opt.verify_cols = *std::max_element(Ws.begin(), Ws.end());
        opt.kv_global_q8 = kv_q8;
        opt.attention_mma = attention_mma;
        fns::Engine engine(model, opt);
        const int NV = model.hp.n_vocab;
        std::fprintf(stderr, "fns-verify: prompt %d, gen %d, ring %d, experts in RAM %d\n", P, n_gen, engine.ring(),
                     engine.placement().host_experts);

        // Context: prefill the prompt except its last token, which is the input of the first step.
        auto fill = [&]() {
            if (P > 1) engine.prefill(prompt.data(), P - 1, 0);
        };
        fill();
        // Warm up the graphs (not timed), then restore the state.
        engine.step_greedy(prompt[P - 1], P - 1);
        for (int w : Ws) {
            std::vector<int32_t> ids(w, prompt[P - 1]), o(w);
            engine.verify(ids.data(), w, P - 1, o.data());
        }
        fill();

        // 1. Sequential reference.
        std::vector<int32_t> ref;
        int32_t tok = prompt[P - 1];
        const auto t0 = Clock::now();
        for (int g = 0; g < n_gen; ++g) {
            tok = engine.step_greedy(tok, P - 1 + g);
            ref.push_back(tok);
        }
        const double t_seq = secs(t0);
        const double t1_ms = 1e3 * t_seq / n_gen;
        std::printf("sequential: %d tokens in %.3f s, %.2f tok/s, T1 %.3f ms\n", n_gen, t_seq, n_gen / t_seq, t1_ms);
        // Input at position p (P-1 .. P-1+n_gen-1): prompt[P-1], then ref.
        auto input_at = [&](int p) { return p == P - 1 ? prompt[P - 1] : ref[p - P]; };

        std::string json = "{\"prompt_tokens\": " + std::to_string(P) + ", \"n_gen\": " + std::to_string(n_gen) +
                           ", \"t1_ms\": " + std::to_string(t1_ms) + ", \"W\": [";
        // 2. Parity and causality, then the cost of one window (same host clock).
        std::vector<float> lseq((size_t) 8 * NV), lver((size_t) 8 * NV), lalt((size_t) 8 * NV);
        for (size_t wi = 0; wi < Ws.size(); ++wi) {
            const int W = Ws[wi];
            double max_abs = 0, kl_sum = 0, kl_max = 0;
            int same = 0, cols_checked = 0, causal_bad = 0, causal_checked = 0;
            const int span = n_gen - W;
            for (int k = 0; k < parity_windows && span > 0; ++k) {
                const int p0 = P - 1 + (int) ((long long) span * k / std::max(1, parity_windows - 1 + (parity_windows == 1)));
                std::vector<int32_t> ids(W), o(W);
                for (int c = 0; c < W; ++c) ids[c] = input_at(p0 + c);
                for (int c = 0; c < W; ++c) engine.step(ids[c], p0 + c, lseq.data() + (size_t) c * NV);
                engine.verify(ids.data(), W, p0, o.data(), lver.data());
                for (int c = 0; c < W; ++c) {
                    const Diff d = compare(lseq.data() + (size_t) c * NV, lver.data() + (size_t) c * NV, NV);
                    max_abs = std::max(max_abs, d.max_abs);
                    kl_sum += d.kl;
                    kl_max = std::max(kl_max, d.kl);
                    same += d.same_argmax;
                    ++cols_checked;
                }
                // Causality: suffix changed from column W/2; columns < W/2 must be bit-identical.
                if (W >= 2) {
                    std::vector<int32_t> alt = ids;
                    for (int c = W / 2; c < W; ++c) alt[c] = (ids[c] + 7919) % NV;
                    engine.verify(alt.data(), W, p0, o.data(), lalt.data());
                    for (int c = 0; c < W / 2; ++c) {
                        ++causal_checked;
                        causal_bad += std::memcmp(lalt.data() + (size_t) c * NV, lver.data() + (size_t) c * NV,
                                                  (size_t) NV * sizeof(float)) != 0;
                    }
                }
                for (int c = 0; c < W; ++c) engine.step(ids[c], p0 + c, nullptr);  // sequential KV restored
            }
            // Window cost: verify repeated at the same position (KV rewritten identically), host clock.
            const int p0 = P - 1;
            std::vector<int32_t> ids(W), o(W);
            for (int c = 0; c < W; ++c) ids[c] = input_at(p0 + c);
            const int reps = 64;
            const auto tv0 = Clock::now();
            for (int r = 0; r < reps; ++r) engine.verify(ids.data(), W, p0, o.data());
            const double tv_ms = 1e3 * secs(tv0) / reps;
            for (int c = 0; c < W; ++c) engine.step(ids[c], p0 + c, nullptr);
        std::printf("W %d : parity %d/%d argmax identical, max|Δ| %.4g, mean KL %.3g max %.3g ; causality %d/%d "
                    "identical columns ; Tv %.3f ms = %.3f T1\n",
                        W, same, cols_checked, max_abs, kl_sum / std::max(1, cols_checked), kl_max,
                        causal_checked - causal_bad, causal_checked, tv_ms, tv_ms / t1_ms);
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "%s{\"W\": %d, \"argmax_same\": %d, \"cols\": %d, \"max_abs\": %.6g, \"kl_mean\": %.6g, "
                          "\"kl_max\": %.6g, \"causal_same\": %d, \"causal_checked\": %d, \"tv_ms\": %.4f}",
                          wi ? ", " : "", W, same, cols_checked, max_abs, kl_sum / std::max(1, cols_checked), kl_max,
                          causal_checked - causal_bad, causal_checked, tv_ms);
            json += buf;
        }
        json += "], \"decode\": [";

        // 3. Greedy speculative decoding.
        bool first_run = true;
        for (const char * kind : {"oracle", "lookup"}) {
            if (draft != "both" && draft != kind) continue;
            const bool oracle = !std::strcmp(kind, "oracle");
            for (int W : Ws) {
                fill();
                std::vector<int32_t> hist(prompt.begin(), prompt.end()), out;
                int pos = P - 1, windows = 0, steps = 0;
                int32_t last = prompt[P - 1];
                std::vector<int32_t> ids(8), o(8), d(8);
                const auto ts = Clock::now();
                while ((int) out.size() < n_gen) {
                    const int room = std::min(W - 1, n_gen - (int) out.size() - 1);
                    int nd = 0;
                    if (room > 0) {
                        if (oracle) {
                            const int g = (int) out.size();
                            nd = std::min(room, n_gen - g);
                            for (int k = 0; k < nd; ++k) d[k] = ref[g + k];  // input of column k + 1
                            // The oracle assumes the reference is already followed; otherwise there is no draft.
                            if (!std::equal(out.begin(), out.end(), ref.begin())) nd = 0;
                        } else {
                            nd = lookup_draft(hist, room, d.data());
                        }
                    }
                    if (nd == 0) {
                        last = engine.step_greedy(last, pos);
                        ++pos;
                        ++steps;
                        out.push_back(last);
                        hist.push_back(last);
                        continue;
                    }
                    ids[0] = last;
                    for (int k = 0; k < nd; ++k) ids[1 + k] = d[k];
                    engine.verify(ids.data(), nd + 1, pos, o.data());
                    ++windows;
                    int c = 0;
                    out.push_back(o[0]);
                    hist.push_back(o[0]);
                    while (c < nd && d[c] == o[c] && (int) out.size() < n_gen) {
                        ++c;
                        out.push_back(o[c]);
                        hist.push_back(o[c]);
                    }
                    last = o[c];
                    pos += c + 1;
                }
                const double t = secs(ts);
                int match = 0;
                while (match < n_gen && out[match] == ref[match]) ++match;
                const double committed = windows ? (double) (n_gen - steps) / windows : 0.0;
                std::printf("decode %s W %d : %.2f tok/s (×%.3f vs sequential), %d windows (%.2f tokens committed "
                            "per window), %d steps ; identical to reference on %d/%d tokens\n",
                            kind, W, n_gen / t, t_seq / t, windows, committed, steps, match, n_gen);
                char buf[384];
                std::snprintf(buf, sizeof buf,
                              "%s{\"draft\": \"%s\", \"W\": %d, \"tok_s\": %.4f, \"speedup\": %.5f, \"windows\": %d, "
                              "\"committed_per_window\": %.4f, \"steps\": %d, \"match\": %d}",
                              first_run ? "" : ", ", kind, W, n_gen / t, t_seq / t, windows, committed, steps, match);
                json += buf;
                first_run = false;
            }
        }
        json += "]}\n";
        if (out_path) {
            FILE * f = std::fopen(out_path, "wb");
            if (!f) throw std::runtime_error(std::string("cannot write: ") + out_path);
            std::fputs(json.c_str(), f);
            std::fclose(f);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fns-verify: %s\n", e.what());
        return 1;
    }
    return 0;
}
