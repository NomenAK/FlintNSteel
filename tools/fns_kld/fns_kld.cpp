// tools/fns_kld/fns_kld.cpp - FlintNSteel quality against a Q8_0 base written by llama-perplexity.
//
//   fns-kld --model F.gguf [--weights gguf|exl3] [--chunks 32]
//           [--base models/kl/q8_0-code.kld] [--ubatch 2048] [-o out.json] [--logits-out path]
//           [--kv-global f16|q8_0]
// --base defaults to FNS_MODELS/kl/q8_0-code.kld when FNS_MODELS is set, else models/kl/q8_0-code.kld
// relative to the working directory.
// --weights exl3: --model is the safetensors directory; the scoring protocol is unchanged.
// --logits-out: first chunk only, raw float32 [n_ctx][n_vocab], no header.
//
// Each chunk restarts at position 0 without BOS (Flint: add_bos_token=false).
// Only the logits at positions n_ctx/2 .. n_ctx-2 are scored; they predict ids[j+1].
// Quantized base, log(p) > -16 threshold, top-token matching and standard errors follow the
// llama.cpp 19ab5a5e protocol (tools/perplexity/perplexity.cpp); CPU scoring is in kld_score.h.
// The reported seconds cover reading probabilities, prefill and scoring, excluding model loading
// and buffer allocation. The JSON output contains statistics accumulated over all chunks.
// With --logits-out, the seconds also include writing the diagnostic file.
#include "fns/engine.h"
#include "fns/model.h"
#include "exl3/exl3_engine.h"
#include "kld_score.h"

#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace {
// Default KL base: FNS_MODELS/kl/q8_0-code.kld, or models/kl/q8_0-code.kld relative to the working directory.
std::string default_base() {
    const char * models = std::getenv("FNS_MODELS");
    if (models && *models) return std::string(models) + "/kl/q8_0-code.kld";
    return "models/kl/q8_0-code.kld";
}

struct Options {
    std::string model;
    std::string weights = "gguf";
    std::string base = default_base();
    std::string output;
    std::string logits_output;
    int chunks = 32;
    int ubatch = 2048;
    bool decode = false;
    bool kv_global_q8 = false;
    bool fa_int8 = false;
    bool fa_swa = false;
    bool attention_mma = false;
};

void usage() {
    std::fprintf(stderr, "usage: fns-kld --model path [--weights gguf|exl3] [--chunks 32] [--base path] "
                         "[--ubatch 2048] [-o out.json] [--logits-out path] [--kv-global f16|q8_0] [--fa-int8 on|off] "
                         "[--fa-swa on|off] [--decode] [--attention-mma]\n");
}

int positive_integer(const char * value) {
    int result = 0;
    const char * end = value + std::strlen(value);
    const auto parsed = std::from_chars(value, end, result);
    if (parsed.ec != std::errc() || parsed.ptr != end || result <= 0) {
        throw std::invalid_argument(std::string("positive integer expected: ") + value);
    }
    return result;
}

// JSON string escaping for paths, including Windows backslashes and control characters.
std::string json_string(const std::string & text) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
                else out << char(c);
        }
    }
    out << '"';
    return out.str();
}

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Each chunk is prefilled with prefill(ids, n, 0, logits). With --decode (ubatch 0) the engine instead steps
// token by token. Scoring does not depend on the engine.
template <class E>
int run(E & engine, fns_kld::Base & base, const Options & opt) {
    std::vector<float> logits(size_t(base.n_ctx()) * base.n_vocab());
    fns_kld::Scorer scorer(base.rows());
    fns_kld::Totals total;
    std::ofstream json;
    if (!opt.output.empty()) {
        json.open(opt.output);
        if (!json) throw std::runtime_error("cannot open output: " + opt.output);
    }
    std::fprintf(stderr, "fns-kld: %d chunks, n_ctx=%d, n_vocab=%d, ubatch=%d, %zu CPU threads\n",
                 opt.chunks, base.n_ctx(), base.n_vocab(), opt.ubatch, scorer.threads());
    std::printf("chunk             PPL(Q)             KL Divergence             Same top p\n");
    const auto start = Clock::now();
    double score_seconds = 0, prefill_seconds = 0;
    for (int chunk = 0; chunk < opt.chunks; ++chunk) {
        base.read_chunk();
        const int32_t * ids = base.tokens(chunk);
        const auto prefill_start = Clock::now();
        // KV is written before attention; the prefill mask hides negative and future positions.
        // Restarting at 0 therefore excludes all KV from the previous chunk, including ring buffers.
        engine.prefill(ids, base.n_ctx(), 0, logits.data());
        prefill_seconds += elapsed(prefill_start);
        if (chunk == 0 && !opt.logits_output.empty()) {
            std::ofstream out(opt.logits_output, std::ios::binary);
            if (!out) throw std::runtime_error("cannot open logits output: " + opt.logits_output);
            out.write(reinterpret_cast<const char *>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
            out.close();
            if (!out) throw std::runtime_error("failed to write logits: " + opt.logits_output);
        }
        const auto score_start = Clock::now();
        scorer.score(base, logits.data() + size_t(base.first()) * base.n_vocab(), ids + base.first(), total);
        score_seconds += elapsed(score_start);
        const auto kl = fns_kld::mean_and_uncertainty(total.sum_kld, total.sum_kld2, total.count);
        const auto nll = fns_kld::mean_and_uncertainty(total.sum_nll, total.sum_nll2, total.count);
        const double ppl = std::exp(nll.first);
        const double top = double(total.n_same_top) / total.count;
        const double top_err = total.count > 1 ? std::sqrt(top * (1 - top) / (total.count - 1)) : 0.;
        std::printf("%4d    %10.4f ± %9.4f    %10.6f ± %9.6f    %7.3f ± %6.3f %%\n",
                    chunk + 1, ppl, ppl * nll.second, kl.first, kl.second, 100 * top, 100 * top_err);
        std::fflush(stdout);
    }
    const double seconds = elapsed(start);
    const auto kl = fns_kld::mean_and_uncertainty(total.sum_kld, total.sum_kld2, total.count);
    const auto nll = fns_kld::mean_and_uncertainty(total.sum_nll, total.sum_nll2, total.count);
    const auto nll_base = fns_kld::mean_and_uncertainty(total.sum_nll_base, total.sum_nll_base2, total.count);
    const double ppl = std::exp(nll.first), ppl_base = std::exp(nll_base.first);
    const double same_top_pct = 100. * total.n_same_top / total.count;
    std::printf("\n====== Perplexity statistics ======\n"
                "Mean PPL(Q)    : %.6f ± %.6f\n"
                "Mean PPL(base) : %.6f ± %.6f\n"
                "Mean KLD       : %.6f ± %.6f\n"
                "Same top p     : %.3f %%\n"
                "Positions      : %llu\n"
                "Seconds        : %.6f (prefill %.3f, score CPU %.3f)\n",
                ppl, ppl * nll.second, ppl_base, ppl_base * nll_base.second,
                kl.first, kl.second, same_top_pct, static_cast<unsigned long long>(total.count),
                seconds, prefill_seconds, score_seconds);
    if (json.is_open()) {
        json << std::setprecision(17)
             << "{\n  \"model\": " << json_string(opt.model)
             << ",\n  \"chunks\": " << opt.chunks
             << ",\n  \"evaluation\": " << json_string(opt.decode ? "decode" : "prefill")
             << ",\n  \"ubatch\": " << opt.ubatch
             << ",\n  \"positions\": " << total.count
             << ",\n  \"kld_mean\": " << kl.first
             << ",\n  \"kld_stderr\": " << kl.second
             << ",\n  \"same_top_pct\": " << same_top_pct
             << ",\n  \"ppl\": " << ppl
             << ",\n  \"ppl_base\": " << ppl_base
             << ",\n  \"seconds\": " << seconds << "\n}\n";
        json.close();
        if (!json) throw std::runtime_error("failed to write JSON output: " + opt.output);
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    Options opt;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            if (arg == "--decode") { opt.decode = true; continue; }
            if (arg == "--attention-mma") { opt.attention_mma = true; continue; }
            if (i + 1 >= argc) { usage(); return 2; }
            const char * value = argv[++i];
            if (arg == "--model") opt.model = value;
            else if (arg == "--weights") opt.weights = value;
            else if (arg == "--base") opt.base = value;
            else if (arg == "--chunks") opt.chunks = positive_integer(value);
            else if (arg == "--ubatch") opt.ubatch = positive_integer(value);
            else if (arg == "-o") opt.output = value;
            else if (arg == "--logits-out") opt.logits_output = value;
            else if (arg == "--kv-global") opt.kv_global_q8 = std::string(value) == "q8_0";
            else if (arg == "--fa-int8") opt.fa_int8 = std::string(value) == "on";
            else if (arg == "--fa-swa") opt.fa_swa = std::string(value) == "on";
            else { usage(); return 2; }
        }
        if (opt.model.empty()) { usage(); return 2; }
        if (opt.weights != "gguf" && opt.weights != "exl3") throw std::invalid_argument("--weights expects gguf or exl3");
        if (opt.decode && opt.weights != "gguf") throw std::invalid_argument("--decode: GGUF only");
        if (opt.attention_mma && opt.weights != "gguf") throw std::invalid_argument("--attention-mma: GGUF only");
        if (opt.decode) opt.ubatch = 0;  // exercises step() at each position, not the prefill flash attention
        fns_kld::Base base(opt.base);
        if (opt.chunks > base.n_chunk()) throw std::invalid_argument("--chunks exceeds the number of chunks in the base");
        if (opt.weights == "exl3") {
            fns::Exl3Model model(opt.model);
            if (model.hp.n_vocab != base.n_vocab()) throw std::runtime_error("model vocabulary differs from the KL base");
            fns::EngineOptions engine_opt;
            engine_opt.n_ctx = base.n_ctx();
            engine_opt.ubatch = opt.ubatch;
            fns::Exl3Engine engine(model, engine_opt);
            return run(engine, base, opt);
        }
        fns::Model model(opt.model);
        if (model.hp.n_vocab != base.n_vocab()) throw std::runtime_error("model vocabulary differs from the KL base");
        fns::EngineOptions engine_opt;
        engine_opt.n_ctx = base.n_ctx();
        engine_opt.ubatch = opt.ubatch;
        engine_opt.kv_global_q8 = opt.kv_global_q8;
        engine_opt.fa_int8 = opt.fa_int8;
        engine_opt.fa_swa = opt.fa_swa;
        engine_opt.attention_mma = opt.attention_mma;
        fns::Engine engine(model, engine_opt);
        return run(engine, base, opt);
    } catch (const std::invalid_argument & e) {
        std::fprintf(stderr, "fns-kld: %s\n", e.what());
        return 2;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fns-kld: %s\n", e.what());
        return 1;
    }
}
