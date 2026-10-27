// tools/fns_bench/fns_bench.cpp - FlintNSteel throughput, same workload as `llama-bench -p 0 -n N -d D` (decode) or
// `llama-bench -p N -n 0 -d D` (prefill).
//
//   fns-bench --model F.gguf [-n 256 | -p N] [-d 0] [-r 1] [-ub N] [--ctx N] [-o jsonl] [--prompt ids.txt]
//             [--weights gguf|exl3] [--expert-profile F] [--expert-vram MiB] [--kv-global f16|q8_0] [--rebalance N]
//             [--rebalance-swap fused|copy] [--fa-int8] [--fa-swa] [--attention-mma]
//
// Fills D positions (synthetic ids as in llama-bench, or --prompt: those ids, D = their count), in batches of -ub
// tokens when -ub is given (otherwise token by token). Decode: 8 untimed warm-up tokens, then N generated tokens
// (GPU argmax), timed. Prefill (-p N, -ub 512 by default): N synthetic tokens at positions D .. D + N - 1, timed.
// Repeats r times on a cleared context.
// jsonl output: one line per repetition, {"n_prompt", "n_gen", "n_depth", "avg_ts", "samples_ts"} as in llama-bench.
#include "fns/engine.h"
#include "fns/model.h"
#include "exl3/exl3_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>

template<class Model, class Engine>
int run(int argc, char ** argv) {
    const char * model_path = nullptr, * prompt_path = nullptr;
    int n_gen = 256, n_prompt = 0, depth = 0, reps = 1;
    bool jsonl = false;
    fns::EngineOptions opt;
    opt.n_ctx = 0;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        const char * v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!std::strcmp(a, "--model") && v) { model_path = v; ++i; }
        else if (!std::strcmp(a, "-n") && v) { n_gen = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "-p") && v) { n_prompt = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "-d") && v) { depth = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "-r") && v) { reps = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--ctx") && v) { opt.n_ctx = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "-ub") && v) { opt.ubatch = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "-o") && v) { jsonl = !std::strcmp(v, "jsonl"); ++i; }
        else if (!std::strcmp(a, "--expert-profile") && v) { opt.expert_profile = v; ++i; }
        else if (!std::strcmp(a, "--expert-vram") && v) { opt.expert_vram = std::atoll(v) << 20; ++i; }
        else if (!std::strcmp(a, "--prompt") && v) { prompt_path = v; ++i; }
        else if (!std::strcmp(a, "--kv-global") && v) { opt.kv_global_q8 = !std::strcmp(v, "q8_0"); ++i; }
        else if (!std::strcmp(a, "--rebalance") && v) { opt.rebalance_every = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--rebalance-swap") && v) {
            if (std::strcmp(v, "fused") && std::strcmp(v, "copy")) {
                std::fprintf(stderr, "fns-bench: --rebalance-swap must be fused or copy\n");
                return 2;
            }
            opt.rebalance_fused_swap = !std::strcmp(v, "fused");
            ++i;
        }
        else if (!std::strcmp(a, "--fa-int8")) { opt.fa_int8 = true; }
        else if (!std::strcmp(a, "--fa-swa")) { opt.fa_swa = true; }
        else if (!std::strcmp(a, "--attention-mma")) { opt.attention_mma = true; }
        else {
            std::fprintf(stderr, "usage: fns-bench --model F.gguf [-n 256 | -p N] [-d 0] [-r 1] [-ub N] [--ctx N] [-o jsonl] "
                                 "[--weights gguf|exl3] [--prompt ids.txt] [--expert-profile F] [--expert-vram MiB] "
                                 "[--kv-global f16|q8_0] [--rebalance N] [--rebalance-swap fused|copy] "
                                 "[--fa-int8] [--fa-swa] [--attention-mma]\n");
            return 2;
        }
    }
    if (!model_path) return 2;
    if (opt.attention_mma && !std::is_same_v<Engine, fns::Engine>) {
        std::fprintf(stderr, "fns-bench: --attention-mma: GGUF only\n");
        return 2;
    }
    if (opt.rebalance_fused_swap && !std::is_same_v<Engine, fns::Engine>) {
        std::fprintf(stderr, "fns-bench: --rebalance-swap fused: GGUF only\n");
        return 2;
    }
    if (n_prompt > 0) n_gen = 0;  // -p: prefill only, like llama-bench -p N -n 0
    if (n_prompt > 0 && opt.ubatch <= 0) opt.ubatch = 512;
    std::vector<int32_t> prompt;
    if (prompt_path) {
        std::ifstream f(prompt_path);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string s = ss.str();
        std::replace(s.begin(), s.end(), ',', ' ');
        std::istringstream in(s);
        long v;
        while (in >> v) prompt.push_back((int32_t) v);
        if (prompt.empty()) {
            std::fprintf(stderr, "fns-bench: no token ids in %s\n", prompt_path);
            return 2;
        }
        depth = (int) prompt.size();
    }
    constexpr int kWarmup = 8;
    if (opt.n_ctx == 0) opt.n_ctx = depth + n_prompt + kWarmup + n_gen + 16;
    try {
        Model model(model_path);
        Engine engine(model, opt);
        if constexpr (std::is_same_v<Engine, fns::Exl3Engine>) {
            size_t free = 0, total = 0;
            const cudaError_t status = cudaMemGetInfo(&free, &total);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            std::fprintf(stderr, "fns-bench EXL3: weights=%zu bytes, VRAM used=%zu free=%zu total=%zu bytes, ring=%d\n",
                         engine.weight_bytes(), total - free, free, total, engine.ring());
        }
        const fns::ExpertPlacement & pl = engine.placement();
        std::fprintf(stderr, "fns-bench: experts in VRAM %d (%.2f GB), in RAM %d (%.2f GB)\n", pl.vram_experts,
                     pl.vram_bytes / 1e9, pl.host_experts, pl.host_bytes / 1e9);
        // Synthetic ids avoid special tokens (like llama-bench's random ids), or use the given prompt.
        auto synth = [](int pos) { return (int32_t) (1000 + (pos * 7919) % 100000); };
        std::vector<int32_t> fill(depth), pp(n_prompt);
        for (int p = 0; p < depth; ++p) fill[p] = prompt.empty() ? synth(p) : prompt[p];
        for (int p = 0; p < n_prompt; ++p) pp[p] = synth(depth + p);
        // Warm-up (untimed): a short prefill loads the MMQ/FA kernels and the ggml pool buffers. Without
        // prompt or depth (decode at d0), one synthetic id is enough.
        const std::vector<int32_t> warm = !pp.empty() ? pp : !fill.empty() ? fill : std::vector<int32_t>{synth(0)};
        if (opt.ubatch > 0) engine.prefill(warm.data(), std::min<int>(64, (int) warm.size()), 0);
        std::vector<double> samples;
        for (int r = 0; r < reps; ++r) {
            int32_t tok = 1000;
            if (depth > 0) tok = engine.prefill(fill.data(), depth, 0);
            if constexpr (std::is_same_v<Engine, fns::Engine>) engine.maybe_rebalance(depth);
            int pos = depth;
            if (n_prompt > 0) {
                const auto t0 = std::chrono::steady_clock::now();
                engine.prefill(pp.data(), n_prompt, pos);
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                samples.push_back(n_prompt / s);
                continue;
            }
            if (prompt.empty()) tok = 1000;
            for (int w = 0; w < kWarmup; ++w, ++pos) tok = engine.step_greedy(tok, pos);
            const auto t0 = std::chrono::steady_clock::now();
            for (int g = 0; g < n_gen; ++g, ++pos) {
                tok = engine.step_greedy(tok, pos);
                if constexpr (std::is_same_v<Engine, fns::Engine>) engine.maybe_rebalance(1);
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            samples.push_back(n_gen / s);
        }
        double avg = 0;
        for (double s : samples) avg += s;
        avg /= samples.size();
        if (jsonl) {
            std::printf("{\"n_prompt\": %d, \"n_gen\": %d, \"n_depth\": %d, \"avg_ts\": %.4f, \"samples_ts\": [", n_prompt,
                        n_gen, depth, avg);
            for (size_t i = 0; i < samples.size(); ++i) std::printf("%s%.4f", i ? ", " : "", samples[i]);
            std::printf("]}\n");
        } else if (n_prompt > 0) {
            std::printf("pp%d @ d%d : %.2f tok/s\n", n_prompt, depth, avg);
        } else {
            std::printf("tg%d @ d%d : %.2f tok/s\n", n_gen, depth, avg);
        }
        if constexpr (std::is_same_v<Engine, fns::Engine>) {
            if (opt.rebalance_every > 0) {
                const auto & placement = engine.placement();
                std::fprintf(stderr, "fns-bench: adaptive cache, %lld swaps", placement.swaps);
                if (placement.rebalance_timing)
                    std::fprintf(stderr, ", %.3f ms, %lld calls, %llu bytes exchanged", placement.rebalance_ms,
                                 placement.rebalance_calls, (unsigned long long) placement.rebalance_bytes);
                std::fprintf(stderr, "\n");
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fns-bench: %s\n", e.what());
        return 1;
    }
    return 0;
}

int main(int argc, char ** argv) {
    bool exl3 = false;
    int n = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--weights") && i + 1 < argc) {
            const char * kind = argv[++i];
            if (std::strcmp(kind, "exl3") && std::strcmp(kind, "gguf")) return 2;
            exl3 = !std::strcmp(kind, "exl3");
        } else argv[n++] = argv[i];
    }
    return exl3 ? run<fns::Exl3Model, fns::Exl3Engine>(n, argv) : run<fns::Model, fns::Engine>(n, argv);
}
