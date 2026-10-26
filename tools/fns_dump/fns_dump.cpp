// tools/fns_dump/fns_dump.cpp - llama.cpp reference logits for model parity checks.
//
// Reads token ids, feeds them to llama.cpp one at a time (like fns-ref: batch 1, token-by-token decode) and writes
// the full logits of each position as float32: file [n_tokens][n_vocab]. Teacher-forcing method adapted from Strata
// (tools/qwen35/q36_llama_dump.cpp); the implementation is FlintNSteel's own.
//
//   fns-dump --model F.gguf --tokens F.txt --out F.f32 [--ngl 99] [--kv f16|q8_0] [--batch 1]
//
// --tokens: ids separated by commas, spaces or newlines.
// --batch N: tokens per llama_decode call (1 = same split as fns-ref; N > 1 = prefill).
// --ngl 0 gives the CPU reference.
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<llama_token> read_ids(const char * path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "fns-dump: cannot open %s\n", path);
        std::exit(2);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    for (char & c : s) {
        if (c == ',') c = ' ';
    }
    std::istringstream in(s);
    std::vector<llama_token> ids;
    long v;
    while (in >> v) ids.push_back((llama_token) v);
    return ids;
}

// --dump-layers F: "l_out-N" layer outputs, [n_tokens][n_layer][n_embd] float32, captured by cb_eval.
// Only with --batch 1 (one row per layer and per batch).
struct LayerDump {
    std::vector<float> buf;
    int n_layer = 0, n_embd = 0;
};

// FNS_DUMP_NODE=name (e.g. "ffn_moe_topk-46"): also prints that tensor to stderr for each batch (diagnostic).
static const char * g_node = std::getenv("FNS_DUMP_NODE");

static bool on_eval(ggml_tensor * t, bool ask, void * user) {
    auto * d = (LayerDump *) user;
    const char * name = ggml_get_name(t);
    const bool is_node = g_node && (std::strcmp(name, g_node) == 0 || (g_node[0] == '*' && std::strstr(name, g_node + 1)));
    if (std::strncmp(name, "l_out-", 6) != 0 && !is_node) return ask ? false : true;
    if (ask) return true;
    if (is_node) {
        const int64_t n = std::min<int64_t>(ggml_nelements(t), 16);
        std::vector<char> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        std::fprintf(stderr, "node %s type %s ne [%lld %lld %lld]:", name, ggml_type_name(t->type),
                     (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
        for (int64_t i = 0; i < n; ++i) {
            if (t->type == GGML_TYPE_I32) std::fprintf(stderr, " %d", ((int32_t *) raw.data())[i]);
            else if (t->type == GGML_TYPE_F32) std::fprintf(stderr, " %.6g", ((float *) raw.data())[i]);
        }
        std::fprintf(stderr, "\n");
        return true;
    }
    const int il = std::atoi(name + 6);
    if (il >= 0 && il < d->n_layer && t->type == GGML_TYPE_F32 && t->ne[0] == d->n_embd) {
        ggml_backend_tensor_get(t, d->buf.data() + (size_t) il * d->n_embd, 0, (size_t) d->n_embd * sizeof(float));
    }
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr, * tokens_path = nullptr, * out_path = nullptr, * layers_path = nullptr;
    int ngl = 99, batch = 1;
    ggml_type kv = GGML_TYPE_F16;
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        const char * v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!std::strcmp(a, "--model") && v) { model_path = v; ++i; }
        else if (!std::strcmp(a, "--tokens") && v) { tokens_path = v; ++i; }
        else if (!std::strcmp(a, "--out") && v) { out_path = v; ++i; }
        else if (!std::strcmp(a, "--ngl") && v) { ngl = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--batch") && v) { batch = std::atoi(v); ++i; }
        else if (!std::strcmp(a, "--kv") && v) { kv = !std::strcmp(v, "q8_0") ? GGML_TYPE_Q8_0 : GGML_TYPE_F16; ++i; }
        else if (!std::strcmp(a, "--dump-layers") && v) { layers_path = v; ++i; }
        else {
            std::fprintf(stderr, "usage: fns-dump --model F.gguf --tokens F.txt --out F.f32 [--ngl 99] [--kv f16|q8_0] [--batch 1]\n");
            return 2;
        }
    }
    if (!model_path || !tokens_path || !out_path || batch < 1 || (layers_path && batch != 1)) {
        std::fprintf(stderr, "fns-dump: --model, --tokens and --out are required\n");
        return 2;
    }
    const std::vector<llama_token> ids = read_ids(tokens_path);
    if (ids.empty()) {
        std::fprintf(stderr, "fns-dump: no token ids in %s\n", tokens_path);
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) ((ids.size() + 255) / 256 * 256 + 256);
    cp.n_batch = cp.n_ubatch = (uint32_t) std::max(batch, 1);
    cp.n_seq_max = 1;
    cp.type_k = cp.type_v = kv;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    LayerDump dump;
    FILE * lout = nullptr;
    if (layers_path || g_node) {
        dump.n_layer = llama_model_n_layer(model);
        dump.n_embd = llama_model_n_embd(model);
        dump.buf.assign((size_t) dump.n_layer * dump.n_embd, 0.0f);
        cp.cb_eval = on_eval;
        cp.cb_eval_user_data = &dump;
    }
    if (layers_path) {
        lout = std::fopen(layers_path, "wb");
        if (!lout) return 1;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * out = std::fopen(out_path, "wb");
    if (!out) {
        std::fprintf(stderr, "fns-dump: cannot write %s\n", out_path);
        return 1;
    }

    llama_batch b = llama_batch_init(batch, 0, 1);
    for (size_t p0 = 0; p0 < ids.size(); p0 += (size_t) batch) {
        const int n = (int) std::min<size_t>((size_t) batch, ids.size() - p0);
        b.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            b.token[j] = ids[p0 + j];
            b.pos[j] = (llama_pos) (p0 + j);
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = 0;
            b.logits[j] = 1;
        }
        if (llama_decode(ctx, b) != 0) {
            std::fprintf(stderr, "fns-dump: llama_decode failed at position %zu\n", p0);
            return 1;
        }
        for (int j = 0; j < n; ++j) {
            std::fwrite(llama_get_logits_ith(ctx, j), sizeof(float), (size_t) n_vocab, out);
        }
        if (lout) std::fwrite(dump.buf.data(), sizeof(float), dump.buf.size(), lout);
    }
    std::fclose(out);
    if (lout) std::fclose(lout);
    std::fprintf(stderr, "fns-dump: %zu positions x %d logits -> %s\n", ids.size(), n_vocab, out_path);

    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
