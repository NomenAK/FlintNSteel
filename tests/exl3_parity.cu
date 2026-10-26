// EXL3 parity: extension outputs for identical weights and inputs, compared bit for bit.
//
// Usage (program name `exl3-parity`):
//   exl3-parity --model <exl3-model> --data <parity-dir>
//       Fixture parity against <parity-dir>/manifest.json (linear, router, MoE). Requires --model and --data.
//   exl3-parity --attention-data <attention-dir>
//       Attention fixtures only; loads no model.
//   exl3-parity --model <exl3-model> --trace-tokens <tokens.i32> --trace-out <new-dir>
//       [--batch 1|8] [--trace-start N] [--trace-count N] [--trace-layer L] [--replay dir]
//       [--reprepare-router] [--public-forward] [--decode] [--lifecycle] [--ring N]
//       Diagnostic trace of the forward pass; the output directory must not exist.
#include "exl3/exl3_model.h"
#include "exl3/exl3_kernels.h"
#include "exl3/exl3_engine.h"
#include "picojson.h"
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#define CK(x)                                                                                                          \
    do {                                                                                                               \
        auto e = (x);                                                                                                  \
        if (e != cudaSuccess)                                                                                          \
            throw std::runtime_error(cudaGetErrorString(e));                                                           \
    } while (0)
struct Arena {
    std::vector<void *> ptrs;
    ~Arena() {
        for (auto p : ptrs)
            cudaFree(p);
    }
    void *alloc(size_t n) {
        void *p;
        CK(cudaMalloc(&p, n));
        ptrs.push_back(p);
        CK(cudaMemset(p, 0, n));
        return p;
    }
    void *upload(const void *h, size_t n) {
        void *p = alloc(n);
        CK(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice));
        return p;
    }
};
std::vector<char> bytes(const std::string &p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("read failed: " + p);
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> b(n);
    if (!f.read(b.data(), n))
        throw std::runtime_error("short read: " + p);
    return b;
}
fns::exl3::Linear linear(Arena &a, const fns::Exl3Model &m, const std::string &key) {
    auto &b = m.tensor(key + ".trellis");
    auto &u = m.tensor(key + ".suh");
    auto &v = m.tensor(key + ".svh");
    return {(uint16_t *)a.upload(b.data, b.bytes),
            (half *)a.upload(u.data, u.bytes),
            (half *)a.upload(v.data, v.bytes),
            (int)b.shape[0] * 16,
            (int)b.shape[1] * 16,
            (int)b.shape[2] / 16};
}
namespace fns {
// Deliberately test-only: the production forward specialization contains no trace work.
class Exl3Diagnostic {
  public:
    struct Capture {
        std::string output, replay;
        int start = 0, count = 0, layer = -2;
        size_t records = 0, replacements = 0;
        Exl3Engine *engine = nullptr;
        bool reprepare = false;
        bool public_forward = false, decode = false, lifecycle = false;
    };
    static void router_initialization(Exl3Engine &engine) {
        Arena arena;
        exl3::Router control;
        control.hi = (signed char *)arena.alloc(2 * 256 * 2048);
        control.scales = (float *)arena.alloc(256 * 4);
        std::vector<char> actual(2 * 256 * 2048), expected(actual.size());
        std::vector<float> actual_scales(256), expected_scales(256);
        for (int layer = 2; layer < 49; ++layer) {
            const auto &router = engine.layers_[layer].router;
            control.weight = router.weight;
            exl3::prepare_router(control, engine.st_);
            CK(cudaStreamSynchronize(engine.st_));
            CK(cudaMemcpy(actual.data(), router.hi, actual.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(expected.data(), control.hi, expected.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(actual_scales.data(), router.scales, 256 * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(expected_scales.data(), control.scales, 256 * 4, cudaMemcpyDeviceToHost));
            if (actual != expected || actual_scales != expected_scales)
                throw std::runtime_error("router initialization, layer " + std::to_string(layer));
        }
        std::printf("EXL3 router initialization : 47/47 quantizations and scales exact, no repair\n");
    }
    static void attention(const std::string &directory) {
        auto raw = bytes(directory + "/manifest.json");
        picojson::value manifest;
        auto error = picojson::parse(manifest, std::string(raw.begin(), raw.end()));
        if (!error.empty()) throw std::runtime_error(error);
        int passed = 0;
        for (const auto &entry : manifest.get<picojson::array>()) {
            Arena arena;
            std::string stem = entry.get("stem").get<std::string>();
            int position = (int)entry.get("position").get<double>();
            int slots = (int)entry.get("slots").get<double>();
            int window = (int)entry.get("window").get<double>();
            int chunks = ((window ? window + 63 : position + 1) + 63) / 64;
            auto upload = [&](const char *suffix) {
                auto data = bytes(directory + "/" + stem + suffix);
                return arena.upload(data.data(), data.size());
            };
            auto q = (float *)upload(".q"), sinks = (float *)upload(".sinks");
            auto k = (half *)upload(".k"), v = (half *)upload(".v");
            auto expected = bytes(directory + "/" + stem + ".out");
            if (expected.size() != 4096 * 4) throw std::runtime_error("attention fixture has unexpected size");
            auto output = (float *)arena.alloc(expected.size());
            auto scratch = (float *)arena.alloc((size_t)32 * chunks * 130 * 4);
            auto pos = (int32_t *)arena.upload(&position, sizeof(position));
            Exl3Engine::decode_attention(q, k, v, scratch, sinks, output, pos, slots, window, chunks, nullptr);
            std::vector<char> actual(expected.size());
            CK(cudaMemcpy(actual.data(), output, actual.size(), cudaMemcpyDeviceToHost));
            if (actual != expected) throw std::runtime_error("attention not bit-exact: " + stem);
            ++passed;
            std::printf("EXL3 attention %s : bit-exact\n", stem.c_str());
        }
        if (passed != 4) throw std::runtime_error("four attention fixtures required");
        std::printf("EXL3 attention : 4/4 exact against Triton, sinks/SWA/global/ring\n");
    }
    static void visit(void *opaque, int position, int layer, const char *name, void *data,
                      int rows, int width, bool fp16, cudaStream_t stream) {
        auto &c = *(Capture *)opaque;
        if (c.layer != -2 && layer != c.layer)
            return;
        for (int r = 0; r < rows; ++r) {
            int p = position + r;
            if (p < c.start || p >= c.start + c.count)
                continue;
            std::string file = "p" + std::to_string(p) + "_l" + std::to_string(layer) + "_" + name + ".f32";
            std::vector<float> host(width);
            std::vector<half> halves(fp16 ? width : 0);
            char *ptr = (char *)data + (size_t)r * width * (fp16 ? 2 : 4);
            CK(cudaMemcpyAsync(fp16 ? (void *)halves.data() : (void *)host.data(), ptr,
                               (size_t)width * (fp16 ? 2 : 4), cudaMemcpyDeviceToHost, stream));
            CK(cudaStreamSynchronize(stream));
            if (fp16)
                for (int i = 0; i < width; ++i)
                    host[i] = __half2float(halves[i]);
            std::ofstream out(c.output + "/" + file, std::ios::binary);
            if (!out.write((const char *)host.data(), width * 4))
                throw std::runtime_error("diagnostic write failed: " + file);
            ++c.records;
            if (!std::strcmp(name, "router_input")) {
                int64_t selected[6];
                half weights[6];
                half score_half[256];
                CK(cudaMemcpyAsync(score_half, c.engine->router_scratch_.scores + r * 256, sizeof(score_half),
                                   cudaMemcpyDeviceToHost, stream));
                CK(cudaMemcpyAsync(selected, c.engine->selected_ + r * 6, sizeof(selected),
                                   cudaMemcpyDeviceToHost, stream));
                CK(cudaMemcpyAsync(weights, c.engine->rw_ + r * 6, sizeof(weights),
                                   cudaMemcpyDeviceToHost, stream));
                CK(cudaStreamSynchronize(stream));
                float score_float[256];
                for (int j = 0; j < 256; ++j)
                    score_float[j] = __half2float(score_half[j]);
                std::ofstream score_file(c.output + "/p" + std::to_string(p) + "_l" +
                                         std::to_string(layer) + "_router_scores.f32", std::ios::binary);
                if (!score_file.write((const char *)score_float, sizeof(score_float)))
                    throw std::runtime_error("diagnostic score write failed");
                for (const char *kind : {"selected", "routing_weights"}) {
                    float values[6];
                    for (int j = 0; j < 6; ++j)
                        values[j] = kind[0] == 's' ? (float)selected[j] : __half2float(weights[j]);
                    std::ofstream routing(c.output + "/p" + std::to_string(p) + "_l" +
                                          std::to_string(layer) + "_" + kind + ".f32", std::ios::binary);
                    if (!routing.write((const char *)values, sizeof(values)))
                        throw std::runtime_error("diagnostic routing write failed");
                    std::string source = c.replay + "/p" + std::to_string(p) + "_l" +
                                         std::to_string(layer) + "_" + kind + ".f32";
                    if (!c.replay.empty() && std::filesystem::exists(source)) {
                        auto replacement = bytes(source);
                        if (replacement.size() != sizeof(values))
                            throw std::runtime_error("routing replay size mismatch");
                        std::memcpy(values, replacement.data(), sizeof(values));
                        if (kind[0] == 's') {
                            for (int j = 0; j < 6; ++j) selected[j] = (int64_t)values[j];
                            CK(cudaMemcpyAsync(c.engine->selected_ + r * 6, selected, sizeof(selected),
                                               cudaMemcpyHostToDevice, stream));
                        } else {
                            for (int j = 0; j < 6; ++j) weights[j] = __float2half_rn(values[j]);
                            CK(cudaMemcpyAsync(c.engine->rw_ + r * 6, weights, sizeof(weights),
                                               cudaMemcpyHostToDevice, stream));
                        }
                        CK(cudaStreamSynchronize(stream));
                        ++c.replacements;
                    }
                }
            }
            if (!c.replay.empty() && std::filesystem::exists(c.replay + "/" + file)) {
                auto input = bytes(c.replay + "/" + file);
                if (input.size() != (size_t)width * 4)
                    throw std::runtime_error("replay size mismatch: " + file);
                std::memcpy(host.data(), input.data(), input.size());
                if (fp16)
                    for (int i = 0; i < width; ++i)
                        halves[i] = __float2half_rn(host[i]);
                CK(cudaMemcpyAsync(ptr, fp16 ? (const void *)halves.data() : (const void *)host.data(),
                                   (size_t)width * (fp16 ? 2 : 4), cudaMemcpyHostToDevice, stream));
                CK(cudaStreamSynchronize(stream));
                ++c.replacements;
            }
        }
    }
    static void run(Exl3Engine &engine, const std::vector<int32_t> &ids, int batch, Capture &capture) {
        capture.engine = &engine;
        router_initialization(engine);
        if (capture.reprepare) {
            CK(cudaStreamSynchronize(nullptr));
            for (int layer = 2; layer < 49; ++layer)
                exl3::prepare_router(engine.layers_[layer].router, engine.st_);
            CK(cudaStreamSynchronize(engine.st_));
        }
        Exl3Engine::Trace trace{&capture, visit};
        std::vector<float> logits(150144);
        for (int p = 0; p < (int)ids.size(); p += capture.decode ? 1 : batch) {
            int rows = capture.decode ? 1 : std::min(batch, (int)ids.size() - p);
            if (capture.public_forward) {
                if (capture.decode)
                    engine.step(ids[p], p, logits.data());
                else {
                    logits.resize((size_t)rows * 150144);
                    engine.prefill(ids.data() + p, rows, p, logits.data());
                }
                for (int row = 0; row < rows; ++row) {
                    std::ofstream file(capture.output + "/p" + std::to_string(p + row) + "_l49_logits.f32", std::ios::binary);
                    if (!file.write((const char *)logits.data() + (size_t)row * 150144 * 4, 150144 * 4))
                        throw std::runtime_error("public logits write failed");
                }
            } else {
                if (capture.decode) {
                    engine.host_io_[0] = ids[p];
                    engine.host_io_[1] = p;
                    CK(cudaMemcpyAsync(engine.io_, engine.host_io_, 8, cudaMemcpyHostToDevice, engine.st_));
                } else
                    CK(cudaMemcpyAsync(engine.tokens_, ids.data() + p, rows * 4, cudaMemcpyHostToDevice, engine.st_));
                engine.forward<true>(rows, capture.decode ? -1 : p, true, nullptr, nullptr, &trace);
                CK(cudaStreamSynchronize(engine.st_));
            }
        }
        if (capture.lifecycle) {
            const int n = (int)ids.size(), back = std::max(0, n - 9);
            std::vector<float> expected((size_t)(n - back) * 150144), again(expected.size());
            for (int p = back; p < n; ++p) {
                auto original = bytes(capture.output + "/p" + std::to_string(p) + "_l49_logits.f32");
                std::memcpy(expected.data() + (size_t)(p - back) * 150144, original.data(), 150144 * 4);
            }
            engine.prefill(ids.data() + back, n - back, back, again.data());
            if (expected != again)
                throw std::runtime_error("backtrack differs from original output");
            std::vector<float> reset((size_t)16 * 150144), reset_again(reset.size());
            for (int p = 0; p < 16; ++p) {
                auto original = bytes(capture.output + "/p" + std::to_string(p) + "_l49_logits.f32");
                std::memcpy(reset.data() + (size_t)p * 150144, original.data(), 150144 * 4);
            }
            engine.prefill(ids.data(), n, 0, nullptr);
            engine.prefill(ids.data(), 16, 0, reset_again.data());
            if (reset != reset_again)
                throw std::runtime_error("reset differs from original output");
            std::printf("EXL3 lifecycle : backtrack %d positions and reset 16 exact\n", n - back);
        }
        std::printf("EXL3 diagnostic : %zu boundaries, %zu replays\n", capture.records, capture.replacements);
    }
};
} // namespace fns

int main(int argc, char **argv) {
    try {
        std::string model, dir;
        std::string token_file, trace_out, replay;
        std::string attention_data;
        int batch = 8, start = 0, count = 0, layer = -2;
        int ring = 1024;
        bool reprepare = false;
        bool public_forward = false, decode = false, lifecycle = false;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--model") && i + 1 < argc)
                model = argv[++i];
            else if (!std::strcmp(argv[i], "--data") && i + 1 < argc)
                dir = argv[++i];
            else if (!std::strcmp(argv[i], "--attention-data") && i + 1 < argc)
                attention_data = argv[++i];
            else if (!std::strcmp(argv[i], "--trace-tokens") && i + 1 < argc)
                token_file = argv[++i];
            else if (!std::strcmp(argv[i], "--trace-out") && i + 1 < argc)
                trace_out = argv[++i];
            else if (!std::strcmp(argv[i], "--replay") && i + 1 < argc)
                replay = argv[++i];
            else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc)
                batch = std::stoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--trace-start") && i + 1 < argc)
                start = std::stoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--trace-count") && i + 1 < argc)
                count = std::stoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--trace-layer") && i + 1 < argc)
                layer = std::stoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--reprepare-router"))
                reprepare = true;
            else if (!std::strcmp(argv[i], "--public-forward"))
                public_forward = true;
            else if (!std::strcmp(argv[i], "--decode"))
                decode = true;
            else if (!std::strcmp(argv[i], "--lifecycle"))
                lifecycle = true;
            else if (!std::strcmp(argv[i], "--ring") && i + 1 < argc)
                ring = std::stoi(argv[++i]);
            else
                return 2;
        }
        if (!attention_data.empty()) {
            fns::Exl3Diagnostic::attention(attention_data);
            return 0;
        }
        if (model.empty()) throw std::runtime_error("--model is required");
        fns::Exl3Model m(model);
        if (!token_file.empty()) {
            auto raw = bytes(token_file);
            if (raw.empty() || raw.size() % 4 || (batch != 1 && batch != 8) || trace_out.empty())
                throw std::runtime_error("diagnostic requires I32 tokens, batch 1 or 8, and a new output directory");
            if (!std::filesystem::create_directory(trace_out))
                throw std::runtime_error("diagnostic directory already exists: " + trace_out);
            std::vector<int32_t> ids(raw.size() / 4);
            std::memcpy(ids.data(), raw.data(), raw.size());
            fns::EngineOptions opt;
            opt.n_ctx = ((int)ids.size() + 255) / 256 * 256;
            opt.ubatch = batch;
            opt.ring = ring;
            fns::Exl3Engine engine(m, opt);
            fns::Exl3Diagnostic::Capture capture{trace_out, replay, start, count ? count : (int)ids.size(), layer};
            capture.reprepare = reprepare;
            capture.public_forward = public_forward;
            capture.decode = decode;
            capture.lifecycle = lifecycle;
            fns::Exl3Diagnostic::run(engine, ids, batch, capture);
            return 0;
        }
        if (dir.empty()) throw std::runtime_error("--data is required");
        fns::exl3::initialize();
        auto manifest = bytes(dir + "/manifest.json");
        picojson::value v;
        auto err = picojson::parse(v, std::string(manifest.begin(), manifest.end()));
        if (!err.empty())
            throw std::runtime_error(err);
        int passed = 0, total = 0;
        for (auto &c : v.get<picojson::array>()) {
            Arena a;
            int rows = (int)c.get("m").get<double>();
            auto stem = c.get("stem").get<std::string>();
            auto xh = bytes(dir + "/" + stem + ".x"), yh = bytes(dir + "/" + stem + ".y");
            auto *x = (half *)a.upload(xh.data(), xh.size());
            void *y = a.alloc(yh.size());
            bool fp32 = true;
            if (c.get("kind").get<std::string>() == "linear") {
                auto w = linear(a, m, c.get("key").get<std::string>());
                fp32 = c.get("fp32").get<bool>();
                auto *had = (half *)a.alloc(xh.size());
                auto *locks = (int *)a.alloc(8 << 20);
                fns::exl3::gemv(w, x, rows, had, y, fp32, locks, nullptr);
            } else if (c.get("kind").get<std::string>() == "router") {
                int layer = (int)c.get("layer").get<double>();
                auto &wt = m.tensor("model.layers." + std::to_string(layer) + ".mlp.gate.weight");
                fns::exl3::Router w;
                w.weight = (half *)a.upload(wt.data, wt.bytes);
                w.hi = (signed char *)a.alloc(2 * 256 * 2048);
                w.scales = (float *)a.alloc(256 * 4);
                fns::exl3::RouterScratch s;
                s.hi = (signed char *)a.alloc(8 * 2048);
                s.lo = (signed char *)a.alloc(8 * 2048);
                s.scales = (float *)a.alloc(8 * 16 * 4);
                s.partial = (float *)a.alloc(8 * 8 * 256 * 4);
                s.scores = (half *)a.alloc(8 * 256 * 2);
                auto *sel = (int64_t *)a.alloc(rows * 6 * 8);
                fns::exl3::prepare_router(w, nullptr);
                fns::exl3::route(w, x, rows, s, sel, (half *)y, nullptr);
                fp32 = false;
                auto ref = bytes(dir + "/" + stem + ".sel");
                std::vector<char> got(ref.size());
                CK(cudaMemcpy(got.data(), sel, got.size(), cudaMemcpyDeviceToHost));
                if (got != ref)
                    throw std::runtime_error("routing indices differ: " + stem);
            } else {
                int layer = (int)c.get("layer").get<double>(), ne = (int)c.get("experts").get<double>();
                std::vector<int64_t> t[9];
                const char *proj[] = {"gate", "up", "down"};
                for (int p = 0; p < 3; ++p)
                    for (int e = 0; e < ne; ++e) {
                        auto w = linear(a, m,
                                        "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) +
                                            "." + proj[p] + "_proj");
                        t[p * 3].push_back((int64_t)w.trellis);
                        t[p * 3 + 1].push_back((int64_t)w.suh);
                        t[p * 3 + 2].push_back((int64_t)w.svh);
                    }
                int64_t *tab[9];
                for (int j = 0; j < 9; ++j)
                    tab[j] = (int64_t *)a.upload(t[j].data(), t[j].size() * 8);
                auto sel = bytes(dir + "/" + stem + ".sel"), rw = bytes(dir + "/" + stem + ".rw");
                MoeCoopParams p = {};
                p.x = x;
                p.x_stride = 2048;
                p.sel = (int64_t *)a.upload(sel.data(), sel.size());
                p.rw = (half *)a.upload(rw.data(), rw.size());
                p.bsz = rows;
                p.topk = 6;
                p.H = p.Hi = p.Ho = p.H_out = 2048;
                p.I = 512;
                p.min_expert = p.max_expert = -1;
                p.n_local = ne;
                p.g_trellis = tab[0];
                p.g_suh = tab[1];
                p.g_svh = tab[2];
                p.u_trellis = tab[3];
                p.u_suh = tab[4];
                p.u_svh = tab[5];
                p.d_trellis = tab[6];
                p.d_suh = tab[7];
                p.d_svh = tab[8];
                p.gated = true;
                p.gu_f32 = c.get("gu_fp32").get<bool>();
                p.slots_max = 48;
                p.rows_max = 8;
                p.had_g = (half *)a.alloc(48 * 2048 * 2);
                p.had_u = (half *)a.alloc(48 * 2048 * 2);
                p.gu_g = a.alloc(48 * 512 * (p.gu_f32 ? 4 : 2));
                p.gu_u = a.alloc(48 * 512 * (p.gu_f32 ? 4 : 2));
                p.act_out = (half *)a.alloc(48 * 512 * 2);
                p.d_out = (float *)a.alloc(48 * 2048 * 4);
                p.ctr_a_len = 48 * 4;
                p.ctr_b_len = 8 * 16;
                p.ctr_a = (int *)a.alloc(exl3_moe_coop_ctr_len(48, 8, 512, 2048) * 4);
                p.ctr_b = p.ctr_a + p.ctr_a_len;
                p.runs = p.ctr_b + p.ctr_b_len;
                p.out = (float *)y;
                p.out_stride = 2048;
                fns::exl3::moe(p, nullptr);
            }
            std::vector<char> got(yh.size());
            CK(cudaMemcpy(got.data(), y, got.size(), cudaMemcpyDeviceToHost));
            bool exact = got == yh;
            float maxerr = 0;
            for (size_t i = 0; i < got.size() / (fp32 ? 4 : 2); ++i) {
                float g = fp32 ? ((float *)got.data())[i] : __half2float(((half *)got.data())[i]);
                float r = fp32 ? ((float *)yh.data())[i] : __half2float(((half *)yh.data())[i]);
                maxerr = std::max(maxerr, std::abs(g - r));
            }
            std::printf("%s : %s max_abs=%.9g\n", stem.c_str(), exact ? "bit-exact" : "FAIL", maxerr);
            ++total;
            if (exact)
                ++passed;
        }
        std::printf("EXL3 : %d/%d bit-exact\n", passed, total);
        return total && passed == total ? 0 : 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "exl3-parity : %s\n", e.what());
        return 1;
    }
}
