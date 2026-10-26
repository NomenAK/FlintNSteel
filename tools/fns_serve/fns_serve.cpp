// tools/fns_serve/fns_serve.cpp - engine process driven by the Python server over stdin/stdout.
//
//   fns-serve --model F.gguf [--ctx 32768] [--ubatch 2048] [--expert-profile F] [--expert-vram MiB]
//             [--tokenizer tokenizer.json --vocab vocab.json] [--spec W] [--spec-sample W]
//
// Protocol: one JSON object per line (stdin: requests, stdout: responses).
//   request   {"tokens": [prompt token ids], "max_tokens": N, "stop": [stop ids],
//             "cache_prompt": true (optional; false forces a full prefill without clearing prompt lookup),
//             "grammar": <xgrammar structural tag, JSON object> (optional),
//             "temperature": t, "top_p": p, "top_k": k, "seed": s, "presence_penalty": x (optional; defaults:
//             greedy, no penalty)}
//   response  {"token": id} for each generated token, as soon as it is known, then
//             {"done": true, "reason": "stop"|"length"|"cancel", "prompt_tokens": n, "reused": r, "generated": g,
//             "prefill_ms": t, "decode_ms": t, "spec_windows": w, "spec_drafted": d,
//             "spec_accepted": a, "spec_emitted": e, "decode_iterations": i}
//   error     {"error": "message"} (the process keeps running)
//   cancel    {"cancel": n}: stops the request whose "id" is n before its next token (reason "cancel").
//             No effect if that request has already finished.
//
// The engine keeps the context of the previous request (prompt + generated tokens). A new request sharing its first
// r tokens recomputes only the rest, provided the sliding ring still holds positions r - 512 .. r - 1, i.e.
// (old length) - r <= ring - 513. Otherwise prefill starts from 0.
// Token choice: argmax, or top-k / top-p / temperature sampling on the GPU (Engine::pick_async), in chained decode:
// the forward pass of the chosen token is queued before the host reads it (Engine::step_next).
// At startup, a warm-up (one prefill batch, one token) runs before {"ready": true}.
// Grammar: with --tokenizer/--vocab and xgrammar linked (FNS_WITH_XGRAMMAR), a "grammar" field is compiled (cached
// by text) and a matcher constrains each token: FillNextTokenBitmask on the CPU, masked argmax on the GPU
// (Engine::pick). The matcher accepts the chosen token and terminates on a stop token.
// Speculation (--spec W, 2 to 8): prompt lookup proposes up to W - 1 tokens; Engine::verify computes the W positions
// in one pass. Column 0's choice is committed, then choices continue while the draft equals the previous choice.
// Column c's choice is that of generated token number draw + c: argmax, or sampling with Philox counter draw + c
// (coupled target uniforms indexed by generation). With a grammar, each column's mask is computed by letting the
// matcher accept the draft, which then rolls back (Rollback) to the committed prefix. Same tokens as without
// speculation for the same seed (bit-identical logits, same sampler, same counter). Without a draft: one step.
// Window width is --spec for greedy and --spec-sample for sampling (default min(2, --spec); 0: no speculation when
// sampling).
// Presence penalty: column c counts d[0 .. c) as already seen, and only committed choices are marked
// (Engine::presence_commit): same state as sequential decoding, same tokens.
#include "fns/lookup.h"
#include "fns/engine.h"
#include "fns/model.h"
#include "exl3/exl3_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "picojson.h"
#ifdef FNS_WITH_XGRAMMAR
#include <xgrammar/xgrammar.h>
#endif

namespace {

std::vector<int32_t> int_array(const picojson::object & o, const char * key) {
    std::vector<int32_t> out;
    const auto it = o.find(key);
    if (it == o.end()) return out;
    if (!it->second.is<picojson::array>()) throw std::runtime_error(std::string("expected array: ") + key);
    for (const auto & v : it->second.get<picojson::array>()) {
        if (!v.is<double>()) throw std::runtime_error(std::string("expected integer in ") + key);
        out.push_back((int32_t) v.get<double>());
    }
    return out;
}

std::string slurp(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot read: ") + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

#ifdef FNS_WITH_XGRAMMAR
// Grammar compiler over Flint's vocabulary. Compiled grammars are cached by structural-tag text, since an agent
// request repeats the same tools on every turn.
struct Grammar {
    std::unique_ptr<xgrammar::GrammarCompiler> compiler;
    std::map<std::string, xgrammar::CompiledGrammar> cache;
    int vocab = 0;

    Grammar(const char * tokenizer_path, const char * vocab_path, const std::vector<int32_t> & stop) {
        picojson::value v;
        const std::string err = picojson::parse(v, slurp(vocab_path));
        if (!err.empty()) throw std::runtime_error("vocab.json : " + err);
        std::vector<std::string> tokens;
        for (const auto & s : v.get<picojson::array>()) tokens.push_back(s.get<std::string>());
        picojson::value md;
        picojson::parse(md, xgrammar::TokenizerInfo::DetectMetadataFromHF(slurp(tokenizer_path)));
        xgrammar::TokenizerInfo info(tokens, (xgrammar::VocabType) (int) md.get("vocab_type").get<double>(),
                                     (int) tokens.size(), stop, md.get("add_prefix_space").get<bool>());
        vocab = info.GetVocabSize();
        compiler = std::make_unique<xgrammar::GrammarCompiler>(info, 8, false);
    }

    const xgrammar::CompiledGrammar & get(const std::string & tag) {
        auto it = cache.find(tag);
        if (it == cache.end()) it = cache.emplace(tag, compiler->CompileStructuralTag(tag)).first;
        return it->second;
    }
};
#endif

void emit(const std::string & line) {
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

// Reads stdin in a thread. Requests are queued; {"cancel": n} is stored at once in `cancel` (the decode loop
// compares it with the request id at each token). Request ids must be unique.
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<long long> cancel{-1};

    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            picojson::value v;
            if (picojson::parse(v, line).empty() && v.is<picojson::object>() && v.contains("cancel") &&
                v.get("cancel").is<double>()) {
                cancel = (long long) v.get("cancel").get<double>();
                continue;
            }
            std::lock_guard<std::mutex> lock(mu);
            lines.push_back(std::move(line));
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lock(mu);
        eof = true;
        cv.notify_one();
    }

    bool take(std::string & line) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return eof || !lines.empty(); });
        if (lines.empty()) return false;
        line = std::move(lines.front());
        lines.pop_front();
        return true;
    }
};

}  // namespace

template<class Model, class Engine>
int run(int argc, char ** argv) {
    const char * model_path = nullptr, * tokenizer_path = nullptr, * vocab_path = nullptr;
    int spec_sample = -1;
    fns::EngineOptions opt;
    opt.n_ctx = 32768;
    opt.ubatch = 2048;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        const char * v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!std::strcmp(a, "--model") && v) { model_path = v; ++i; }
        else if (!std::strcmp(a, "--ctx") && v) { opt.n_ctx = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--ubatch") && v) { opt.ubatch = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--expert-profile") && v) { opt.expert_profile = v; ++i; }
        else if (!std::strcmp(a, "--expert-vram") && v) { opt.expert_vram = std::atoll(v) << 20; ++i; }
        else if (!std::strcmp(a, "--tokenizer") && v) { tokenizer_path = v; ++i; }
        else if (!std::strcmp(a, "--vocab") && v) { vocab_path = v; ++i; }
        else if (!std::strcmp(a, "--spec") && v) { opt.verify_cols = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--spec-sample") && v) { spec_sample = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--kv-global") && v) { opt.kv_global_q8 = !std::strcmp(v, "q8_0"); ++i; }
        else if (!std::strcmp(a, "--fa-int8")) { opt.fa_int8 = true; }
        else if (!std::strcmp(a, "--fa-swa")) { opt.fa_swa = true; }
        else if (!std::strcmp(a, "--attention-mma")) { opt.attention_mma = true; }
        else if (!std::strcmp(a, "--rebalance") && v) { opt.rebalance_every = std::atoi(v); ++i; }
        else {
            std::fprintf(stderr, "usage: fns-serve --model F.gguf [--ctx 32768] [--ubatch 2048] [--expert-profile F] "
                                 "[--weights gguf|exl3] [--expert-vram MiB] [--tokenizer tokenizer.json --vocab vocab.json] "
                                 "[--spec W] [--spec-sample W] [--kv-global f16|q8_0] [--fa-int8] [--fa-swa] [--attention-mma] [--rebalance N]\n");
            return 2;
        }
    }
    if (!model_path) return 2;
    if (opt.attention_mma && !std::is_same_v<Engine, fns::Engine>) {
        std::fprintf(stderr, "fns-serve: --attention-mma: GGUF only\n");
        return 2;
    }
    // Window width for greedy (--spec) and sampling (--spec-sample, default min(2, --spec)). The best width differs,
    // since sampled drafts are accepted less often at temperature 1.
    const int spec = opt.verify_cols;
    if (spec_sample < 0) spec_sample = std::min(spec, 2);
    if (spec == 1 || spec < 0 || spec > 8 || spec_sample == 1 || spec_sample > spec) {
        std::fprintf(stderr, "fns-serve: --spec must be 2 to 8, --spec-sample 0 or 2 to --spec\n");
        return 2;
    }
    try {
        if (spec > 0 && !std::is_same_v<Engine, fns::Engine>) throw std::runtime_error("--spec: GGUF only");
        Model model(model_path);
        Engine engine(model, opt);
        const int ring = engine.ring();
#ifdef FNS_WITH_XGRAMMAR
        std::unique_ptr<Grammar> grammar;
        // Matcher stop token: <|end_of_text|> only, since the structural-tag generator requires it to be the tokenizer
        // EOS. <|user|> and <|observation|> are turn ends that the grammar produces itself; declared as stops, they
        // could no longer be generated by the structural tag (repeated tool calls would never end).
        constexpr int32_t kEndOfText = 150001;  // <|end_of_text|> (Flint tokenizer.json)
        if (tokenizer_path && vocab_path)
            grammar = std::make_unique<Grammar>(tokenizer_path, vocab_path, std::vector<int32_t>{kEndOfText});
        const bool has_grammar = grammar != nullptr;
#else
        const bool has_grammar = false;
#endif
        // Warm-up: a full prefill batch loads the MMQ/FlashAttention kernels, cuBLAS and the ggml pool; one decode
        // token captures the graph. The KV written here is not reused (empty context below), so the first request
        // starts at position 0.
        {
            std::vector<int32_t> warm(std::min(opt.ubatch > 0 ? opt.ubatch : 1, opt.n_ctx - 2), 1000);
            engine.prefill(warm.data(), (int) warm.size(), 0);
            engine.pick_async(nullptr);
            engine.step_next((int) warm.size());
            engine.pick_wait();
            engine.pick(nullptr);  // also waits for the forward launched by step_next
            if constexpr (std::is_same_v<Engine, fns::Engine>) {
                for (int w = 2; w <= spec; ++w) {  // capture verify graphs; the KV is discarded with the rest
                    std::vector<int32_t> ids(w, 1000), out(w);
                    engine.verify(ids.data(), w, 0, out.data());
                }
            }
        }

        std::fprintf(stderr, "fns-serve: ready (context %d, batch %d, ring %d, grammar %s, speculation %d)\n",
                     opt.n_ctx, opt.ubatch, ring, has_grammar ? "yes" : "no", spec);
        emit(std::string("{\"ready\": true, \"grammar\": ") + (has_grammar ? "true" : "false") + "}");
        std::vector<int32_t> ctx;  // tokens whose KV is in the cache, positions 0 .. ctx.size() - 1
        Input input;
        std::thread reader([&] { input.run(); });
        reader.detach();  // blocked in getline; the process exits with main
        std::string line;
        while (input.take(line)) {
            try {
                picojson::value req;
                const std::string perr = picojson::parse(req, line);
                if (!perr.empty() || !req.is<picojson::object>()) throw std::runtime_error("invalid JSON request");
                const picojson::object & o = req.get<picojson::object>();
                const std::vector<int32_t> prompt = int_array(o, "tokens");
                const std::vector<int32_t> stop = int_array(o, "stop");
                const auto mt = o.find("max_tokens");
                const int max_tokens = mt != o.end() && mt->second.is<double>() ? (int) mt->second.get<double>() : 256;
                const auto cp = o.find("cache_prompt");
                if (cp != o.end() && !cp->second.is<bool>())
                    throw std::runtime_error("cache_prompt must be a boolean");
                const bool cache_prompt = cp == o.end() || cp->second.get<bool>();
                if (prompt.empty()) throw std::runtime_error("empty prompt");
                if ((int) prompt.size() + max_tokens > opt.n_ctx) throw std::runtime_error("context exceeded");
                for (int32_t t : prompt)
                    if (t < 0 || t >= model.hp.n_vocab) throw std::runtime_error("token outside the vocabulary");
                // Sampling: "temperature", "top_p", "top_k", "seed", "presence_penalty" (defaults: greedy, no penalty).
                // The draw for generated token n uses Philox counter n, so the same request replays the same tokens.
                fns::SampleParams sp;
                auto num = [&](const char * k, double def) {
                    const auto it = o.find(k);
                    return it != o.end() && it->second.is<double>() ? it->second.get<double>() : def;
                };
                sp.temperature = (float) num("temperature", 0.0);
                sp.top_p = (float) num("top_p", 1.0);
                sp.top_k = (int) num("top_k", 0.0);
                sp.seed = (uint64_t) num("seed", 0.0);
                sp.presence_penalty = (float) num("presence_penalty", 0.0);
                if (sp.presence_penalty != 0.0f) engine.reset_presence();
                uint64_t draw = 0;
#ifdef FNS_WITH_XGRAMMAR
                std::unique_ptr<xgrammar::GrammarMatcher> matcher;
                std::vector<int32_t> mask;
                DLTensor mt_tensor{};
                int64_t mask_shape[1] = {0};
                const auto g = o.find("grammar");
                if (g != o.end() && !g->second.is<picojson::null>()) {
                    if (!grammar) throw std::runtime_error("grammar requested but fns-serve was started without --tokenizer/--vocab");
                    matcher = std::make_unique<xgrammar::GrammarMatcher>(grammar->get(g->second.serialize()), std::nullopt,
                                                                         false, spec > 0 ? spec : -1);
                    mask.assign(xgrammar::GetBitmaskSize(grammar->vocab), 0);
                    mask_shape[0] = (int64_t) mask.size();
                    mt_tensor.data = mask.data();
                    mt_tensor.device = DLDevice{kDLCPU, 0};
                    mt_tensor.ndim = 1;
                    mt_tensor.dtype = xgrammar::GetBitmaskDLType();
                    mt_tensor.shape = mask_shape;
                }
                // Next-token choice from the current logits, masked by the grammar if any, in two steps (chained
                // decode, Engine::pick_async): begin queues the choice, end waits for it and validates it.
                auto begin_choice = [&]() {
                    if (!matcher) return engine.pick_async(nullptr, sp, draw++);
                    matcher->FillNextTokenBitmask(&mt_tensor);
                    engine.pick_async((const uint32_t *) mask.data(), sp, draw++);
                };
                auto end_choice = [&]() -> int32_t {
                    const int32_t t = engine.pick_wait();
                    if (matcher && (t < 0 || !matcher->AcceptToken(t)))
                        throw std::runtime_error("token rejected by the grammar");
                    return t;
                };
                // Masks for the columns of a window: column c's mask follows the committed prefix + d[0 .. c).
                // Returns the number of usable columns (a draft rejected by the grammar cuts the window) and leaves
                // the matcher at the committed prefix.
                std::vector<int32_t> wmask;
                auto window_masks = [&](const int32_t * d, int nd) -> int {
                    if (!matcher) return nd + 1;
                    const size_t words = mask.size();
                    wmask.resize((size_t) (nd + 1) * words);
                    int c = 0, accepted = 0;
                    for (;; ++c) {
                        matcher->FillNextTokenBitmask(&mt_tensor);
                        std::copy(mask.begin(), mask.end(), wmask.begin() + (size_t) c * words);
                        if (c == nd || !matcher->AcceptToken(d[c])) break;
                        ++accepted;
                        if (matcher->IsTerminated()) break;  // d[c] terminates: nothing to predict after it
                    }
                    if (accepted) matcher->Rollback(accepted);
                    return std::min(c, nd) + 1;
                };
#else
                if (o.count("grammar") && !o.at("grammar").is<picojson::null>())
                    throw std::runtime_error("fns-serve was built without xgrammar");
                auto begin_choice = [&]() { engine.pick_async(nullptr, sp, draw++); };
                auto end_choice = [&]() -> int32_t { return engine.pick_wait(); };
                auto window_masks = [&](const int32_t *, int nd) -> int { return nd + 1; };
                const bool matcher = false;
#endif
                // Common prefix, keeping at least one token to compute (the logits of the last prompt token).
                size_t r = 0;
                if (cache_prompt)
                    while (r < ctx.size() && r + 1 < prompt.size() && ctx[r] == prompt[r]) ++r;
                if ((long) ctx.size() - (long) r > ring - 513) r = 0;
                // Chained decode: the forward pass of the chosen token is queued (step_next) before the host reads it;
                // the grammar mask of the next token is computed during that forward. When the token turns out to be a
                // stop, this forward has written the KV at position ctx.size(); the reuse rule above (ring - 513)
                // already leaves that position as margin in the ring.
                const auto t0 = std::chrono::steady_clock::now();
                engine.prefill(prompt.data() + r, (int) (prompt.size() - r), (int) r);
                if constexpr (std::is_same_v<Engine, fns::Engine>) engine.maybe_rebalance((int) (prompt.size() - r));
                ctx.assign(prompt.begin(), prompt.end());
                const long long id = (long long) num("id", -1.0);
                int generated = 0, windows = 0, drafted = 0, accepted = 0, spec_emitted = 0, iterations = 0;
                const char * reason = "length";
                // Speculation: greedy, or coupled sampling (each column samples with the Philox counter of its
                // generation index, giving the same tokens as sequential decoding for the same seed). Presence
                // penalty: column c counts draft d[0 .. c) as already seen (Engine::verify), then only committed
                // choices are marked (presence_commit): same state as sequential decoding.
                const bool speculate = (sp.temperature <= 0.0f ? spec : spec_sample) > 0;
                std::chrono::steady_clock::time_point t1;
                if (!speculate) {
                    begin_choice();
                    if (max_tokens > 1) engine.step_next((int) ctx.size());
                    int32_t tok = end_choice();
                    t1 = std::chrono::steady_clock::now();
                    while (generated < max_tokens) {
                        emit("{\"token\": " + std::to_string(tok) + "}");
                        ++generated;
                        if (std::find(stop.begin(), stop.end(), tok) != stop.end()) { reason = "stop"; break; }
                        if (generated == max_tokens) break;
                        if (id >= 0 && input.cancel == id) { reason = "cancel"; break; }
                        ctx.push_back(tok);  // its forward is already queued (position ctx.size() - 1)
                        ++iterations;  // one post-first-token decode, including the final choice
                        begin_choice();
                        // Adaptive cache: between the choice (queued) and the next forward. The swap is queued behind
                        // the forward already launched, so the next forward sees the new table.
                        if constexpr (std::is_same_v<Engine, fns::Engine>) engine.maybe_rebalance(1);
                        if (generated + 1 < max_tokens) engine.step_next((int) ctx.size());
                        tok = end_choice();
                    }
                } else if constexpr (std::is_same_v<Engine, fns::Engine>) {
                    // Speculation. `last`: last committed token, not yet consumed by a forward (its KV is written at
                    // ctx.size() - 1 by the next window). The first choice comes from prefill (counter 0).
                    begin_choice();
                    int32_t last = end_choice();
                    t1 = std::chrono::steady_clock::now();
                    std::vector<int32_t> ids(spec), out(spec), d(spec);
                    auto commit = [&](int32_t t) {  // emits t; returns true if generation stops
                        emit("{\"token\": " + std::to_string(t) + "}");
                        ++generated;
                        ctx.push_back(t);
                        if (std::find(stop.begin(), stop.end(), t) != stop.end()) { reason = "stop"; return true; }
                        if (generated == max_tokens) return true;
                        if (id >= 0 && input.cancel == id) { reason = "cancel"; return true; }
                        return false;
                    };
                    bool done = commit(last);
                    while (!done) {
                        // ctx ends with `last`; the window starts at its position, ctx.size() - 1.
                        const int width = sp.temperature <= 0.0f ? spec : spec_sample;
                        const int room = std::min(width - 1, max_tokens - generated);
                        int nd = room > 0 ? fns::lookup_draft(ctx, room, d.data()) : 0;
                        const int W = nd > 0 ? window_masks(d.data(), nd) : 1;
                        nd = W - 1;
                        const int p0 = (int) ctx.size() - 1;
                        if constexpr (std::is_same_v<Engine, fns::Engine>) engine.maybe_rebalance(1);
                        ++iterations;  // a no-draft step or one verification call
                        if (nd == 0) {
                            engine.step_logits(last, p0);
                            begin_choice();
                            last = end_choice();
                            done = commit(last);
                            continue;
                        }
                        ids[0] = last;
                        for (int k = 0; k < nd; ++k) ids[1 + k] = d[k];
                        // Column c: choice for generated token number `draw + c` (draw = tokens already drawn).
                        engine.verify(ids.data(), W, p0, out.data(), nullptr,
                                      matcher ? (const uint32_t *) wmask.data() : nullptr, sp, draw);
                        ++windows;
                        drafted += nd;
                        // Column c: choice after ids[0 .. c]. Valid while the previous inputs were committed.
                        int engaged = 0;
                        for (int c = 0; c < W && !done; ++c) {
                            const int32_t t = out[c];
#ifdef FNS_WITH_XGRAMMAR
                            if (matcher && (t < 0 || !matcher->AcceptToken(t)))
                                throw std::runtime_error("token rejected by the grammar");
#endif
                            last = t;
                            ++draw;
                            ++engaged;
                            ++spec_emitted;
                            if (c < nd && t == d[c]) ++accepted;  // count even when this token stops or fills the budget
                            done = commit(t);
                            if (c == nd || t != d[c]) break;  // next column was computed on another token
                        }
                        if (sp.presence_penalty != 0.0f) engine.presence_commit(engaged);
                        engine.maybe_rebalance(engaged - 1);
                    }
                    ctx.pop_back();  // the last committed token has no KV (as in the chained path)
                }
                const auto t2 = std::chrono::steady_clock::now();
                char buf[512];
                std::snprintf(buf, sizeof buf,
                              "{\"done\": true, \"reason\": \"%s\", \"prompt_tokens\": %zu, \"reused\": %zu, "
                              "\"generated\": %d, \"prefill_ms\": %.1f, \"decode_ms\": %.1f, \"spec_windows\": %d, "
                              "\"spec_drafted\": %d, \"spec_accepted\": %d, \"spec_emitted\": %d, \"decode_iterations\": %d}",
                              reason, prompt.size(), r, generated,
                              std::chrono::duration<double, std::milli>(t1 - t0).count(),
                              std::chrono::duration<double, std::milli>(t2 - t1).count(), windows, drafted,
                              accepted, spec_emitted, iterations);
                emit(buf);
            } catch (const std::exception & e) {
                ctx.clear();  // unknown state: the next request restarts from 0
                std::string msg = e.what();
                std::replace(msg.begin(), msg.end(), '"', '\'');
                emit("{\"error\": \"" + msg + "\"}");
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fns-serve: %s\n", e.what());
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
