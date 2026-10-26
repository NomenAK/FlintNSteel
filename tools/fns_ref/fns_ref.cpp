// tools/fns_ref/fns_ref.cpp - FlintNSteel reference command-line path (analogous to strata-q36ref).
//
//   fns-ref --model F.gguf --tokens F.txt [--out F.f32] [--gen N] [--ctx N] [--ring N] [--ubatch N]
//           [--expert-profile F] [--expert-vram MiB] [--profile-out F] [--route-trace F] [--out-last N]
//           [--prefill-last N] [--kv-global f16|q8_0] [--fa-int8] [--fa-swa] [--attention-mma]
//
// --out-last N (with --ubatch): batched prefill of all but the last N positions, then those N token by token;
// --out receives only their logits. --prefill-last N: same, but the last N also go through prefill.
//
// --ubatch N: feeds the ids in batches of N tokens (prefill path: MMQ and FlashAttention) instead of one by one.
//
// --profile-out F: counts routed experts per layer and writes [n_layer][n_expert] uint32 (routing profile
// for the expert cache, read by --expert-profile).
//
// --route-trace F: writes, for each position (token by token), the experts routed in each layer:
// uint8 [n_tokens][n_layer][n_expert], 1 if selected (expert preload oracle).
//
// Feeds the ids from --tokens one at a time (teacher forcing) and writes each position's logits as float32,
// [n_tokens][n_vocab], same format as fns-dump (compare with bench/scripts/parity.py). --gen N then generates N
// greedy tokens and prints them. Without --out, only timing and greedy output are shown.
#include "fns/engine.h"
#include "fns/model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int32_t> read_ids(const char * path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "fns-ref: cannot open %s\n", path);
        std::exit(2);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream in(s);
    std::vector<int32_t> ids;
    long v;
    while (in >> v) ids.push_back((int32_t) v);
    return ids;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr, * tokens_path = nullptr, * out_path = nullptr, * layers_path = nullptr;
    const char * profile_out = nullptr, * route_trace = nullptr;
    int gen = 0, out_last = 0, prefill_last = 0;
    fns::EngineOptions opt;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        const char * v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!std::strcmp(a, "--model") && v) { model_path = v; ++i; }
        else if (!std::strcmp(a, "--tokens") && v) { tokens_path = v; ++i; }
        else if (!std::strcmp(a, "--out") && v) { out_path = v; ++i; }
        else if (!std::strcmp(a, "--gen") && v) { gen = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--ctx") && v) { opt.n_ctx = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--ring") && v) { opt.ring = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--dump-layers") && v) { layers_path = v; ++i; }
        else if (!std::strcmp(a, "--expert-profile") && v) { opt.expert_profile = v; ++i; }
        else if (!std::strcmp(a, "--expert-vram") && v) { opt.expert_vram = std::atoll(v) << 20; ++i; }
        else if (!std::strcmp(a, "--profile-out") && v) { profile_out = v; opt.count_routing = true; ++i; }
        else if (!std::strcmp(a, "--route-trace") && v) { route_trace = v; opt.count_routing = true; ++i; }
        else if (!std::strcmp(a, "--out-last") && v) { out_last = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--prefill-last") && v) { prefill_last = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--kv-global") && v) { opt.kv_global_q8 = !std::strcmp(v, "q8_0"); ++i; }
        else if (!std::strcmp(a, "--fa-int8")) { opt.fa_int8 = true; }
        else if (!std::strcmp(a, "--fa-swa")) { opt.fa_swa = true; }
        else if (!std::strcmp(a, "--attention-mma")) { opt.attention_mma = true; }
        else if (!std::strcmp(a, "--ubatch") && v) { opt.ubatch = std::atoi(v); ++i; }
        else {
            std::fprintf(stderr, "usage: fns-ref --model F.gguf --tokens F.txt [--out F.f32] [--gen N] [--ctx N] [--ring N]\n");
            return 2;
        }
    }
    if (!model_path || !tokens_path) {
        std::fprintf(stderr, "fns-ref: --model and --tokens are required\n");
        return 2;
    }
    try {
        const std::vector<int32_t> ids = read_ids(tokens_path);
        if (ids.empty()) throw std::runtime_error("no token ids");
        opt.n_ctx = std::max<int>(opt.n_ctx, (int) ids.size() + gen + 1);

        const auto t0 = std::chrono::steady_clock::now();
        fns::Model model(model_path);
        fns::Engine engine(model, opt);
        const auto t1 = std::chrono::steady_clock::now();
        const fns::ExpertPlacement & pl = engine.placement();
        std::fprintf(stderr, "fns-ref: %.2f GB of weights in VRAM, load %.1f s; experts in VRAM %d, in RAM %d (%.2f GB)\n",
                     engine.weight_bytes() / 1e9, std::chrono::duration<double>(t1 - t0).count(), pl.vram_experts,
                     pl.host_experts, pl.host_bytes / 1e9);

        const int n_vocab = model.hp.n_vocab;
        std::vector<float> logits((size_t) n_vocab);
        FILE * out = out_path ? std::fopen(out_path, "wb") : nullptr;
        if (out_path && !out) throw std::runtime_error(std::string("cannot write: ") + out_path);
        FILE * lout = layers_path ? std::fopen(layers_path, "wb") : nullptr;
        if (layers_path && !lout) throw std::runtime_error(std::string("cannot write: ") + layers_path);
        std::vector<float> layer_buf(lout ? (size_t) model.hp.n_layer * model.hp.n_embd : 0);

        auto argmax = [&]() { return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin()); };
        const auto t2 = std::chrono::steady_clock::now();
        if (opt.ubatch > 0 && !lout && prefill_last > 0) {
            // Prefill all but the last N positions, then prefill those N with their logits: the prefill path (MMQ,
            // FlashAttention) at full depth, without writing [n][n_vocab].
            const int n = (int) ids.size(), head = std::max(0, n - prefill_last);
            if (head > 0) engine.prefill(ids.data(), head, 0);
            std::vector<float> last((size_t) (n - head) * n_vocab);
            engine.prefill(ids.data() + head, n - head, head, last.data());
            if (out) std::fwrite(last.data(), sizeof(float), last.size(), out);
        } else if (opt.ubatch > 0 && !lout && out_last > 0) {
            // Batched prefill of positions 0 .. n - N - 1, then the last N token by token: logits for those N
            // positions only (deep context without writing [n][n_vocab]).
            const int n = (int) ids.size(), head = std::max(0, n - out_last);
            if (head > 0) engine.prefill(ids.data(), head, 0);
            for (int p = head; p < n; ++p) {
                engine.step(ids[p], p, logits.data());
                if (out) std::fwrite(logits.data(), sizeof(float), logits.size(), out);
            }
        } else if (opt.ubatch > 0 && !lout) {
            // Batched prefill: same per-position logits, computed in batches of --ubatch tokens.
            std::vector<float> all(out ? ids.size() * (size_t) n_vocab : 0);
            engine.prefill(ids.data(), (int) ids.size(), 0, out ? all.data() : nullptr);
            if (out) std::fwrite(all.data(), sizeof(float), all.size(), out);
            if (out) std::copy(all.end() - n_vocab, all.end(), logits.begin());
        } else {
            FILE * rt = route_trace ? std::fopen(route_trace, "wb") : nullptr;
            if (route_trace && !rt) throw std::runtime_error(std::string("cannot write: ") + route_trace);
            std::vector<uint32_t> total;
            std::vector<uint8_t> sel;
            for (size_t p = 0; p < ids.size(); ++p) {
                engine.step(ids[p], (int) p, logits.data(), lout ? layer_buf.data() : nullptr);
                if (out) std::fwrite(logits.data(), sizeof(float), logits.size(), out);
                if (lout) std::fwrite(layer_buf.data(), sizeof(float), layer_buf.size(), lout);
                if (rt) {  // single-token counts: 1 per selected expert; running total kept for --profile-out
                    const std::vector<uint32_t> c = engine.routing_counts();
                    engine.reset_routing_counts();
                    if (total.empty()) total.assign(c.size(), 0);
                    sel.resize(c.size());
                    for (size_t i = 0; i < c.size(); ++i) {
                        total[i] += c[i];
                        sel[i] = c[i] ? 1 : 0;
                    }
                    std::fwrite(sel.data(), 1, sel.size(), rt);
                }
            }
            if (rt) std::fclose(rt);
            if (rt && profile_out) {
                FILE * pf = std::fopen(profile_out, "wb");
                if (!pf) throw std::runtime_error(std::string("cannot write: ") + profile_out);
                std::fwrite(total.data(), sizeof(uint32_t), total.size(), pf);
                std::fclose(pf);
                profile_out = nullptr;
            }
        }
        if (lout) std::fclose(lout);
        const auto t3 = std::chrono::steady_clock::now();
        if (out) std::fclose(out);
        std::fprintf(stderr, "fns-ref: %zu forced positions, %.1f tok/s\n", ids.size(),
                     ids.size() / std::chrono::duration<double>(t3 - t2).count());
        if (profile_out) {
            const std::vector<uint32_t> c = engine.routing_counts();
            FILE * pf = std::fopen(profile_out, "wb");
            if (!pf) throw std::runtime_error(std::string("cannot write: ") + profile_out);
            std::fwrite(c.data(), sizeof(uint32_t), c.size(), pf);
            std::fclose(pf);
        }

        if (gen > 0) {
            int32_t tok = argmax();
            std::printf("gen:");
            const auto t4 = std::chrono::steady_clock::now();
            for (int g = 0; g < gen; ++g) {
                std::printf(" %d", tok);
                if (std::find(model.eog_ids.begin(), model.eog_ids.end(), tok) != model.eog_ids.end()) break;
                engine.step(tok, (int) (ids.size() + g), logits.data());
                tok = argmax();
            }
            const auto t5 = std::chrono::steady_clock::now();
            std::printf("\n");
            std::fprintf(stderr, "fns-ref: greedy %.1f tok/s\n", gen / std::chrono::duration<double>(t5 - t4).count());
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fns-ref: %s\n", e.what());
        return 1;
    }
    return 0;
}
