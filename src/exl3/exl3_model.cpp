#include "exl3_model.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "picojson.h"
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <limits>
#include <cstring>
namespace fns {
namespace {
picojson::value json(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot read: " + path);
    picojson::value v;
    std::string err = picojson::parse(v, in);
    if (!err.empty() || !v.is<picojson::object>())
        throw std::runtime_error("invalid JSON: " + path + " " + err);
    return v;
}
} // namespace
struct Exl3Model::Mapping {
    HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr;
    const uint8_t *data = nullptr;
    size_t size = 0;
    explicit Mapping(const std::string &path) {
        file = CreateFileW(std::filesystem::path(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
            throw std::runtime_error("cannot open: " + path);
        LARGE_INTEGER length;
        if (!GetFileSizeEx(file, &length) || length.QuadPart < 8) {
            CloseHandle(file);
            throw std::runtime_error("safetensors file too short: " + path);
        }
        size = (size_t)length.QuadPart;
        mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping)
            data = (const uint8_t *)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!data) {
            if (mapping)
                CloseHandle(mapping);
            CloseHandle(file);
            throw std::runtime_error("cannot map file: " + path);
        }
    }
    ~Mapping() {
        if (data)
            UnmapViewOfFile(data);
        if (mapping)
            CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE)
            CloseHandle(file);
    }
};
Exl3Model::~Exl3Model() = default;
Exl3Model::Exl3Model(const std::string &directory) {
    auto c = json(directory + "/config.json");
    if (c.get("model_type").get<std::string>() != "flint")
        throw std::runtime_error("EXL3: model is not Flint");
    auto n = [&](const char *key) { return (int)c.get(key).get<double>(); };
    hp.n_layer = n("num_hidden_layers");
    hp.n_embd = n("hidden_size");
    hp.n_head = n("num_attention_heads");
    hp.n_head_kv = n("num_key_value_heads");
    hp.head_dim = n("head_dim");
    hp.n_ff = n("intermediate_size");
    hp.n_ff_exp = n("moe_intermediate_size");
    hp.n_ff_shexp = n("shared_expert_intermediate_size");
    hp.n_expert = n("num_experts");
    hp.n_expert_used = n("num_experts_per_tok");
    hp.n_dense_lead = 2;
    hp.n_swa = n("sliding_window");
    hp.n_vocab = n("vocab_size");
    hp.n_ctx_train = n("max_position_embeddings");
    hp.rms_eps = (float)c.get("rms_norm_eps").get<double>();
    hp.rope_base = (float)c.get("rope_theta").get<double>();
    hp.expert_weights_norm = true;
    hp.expert_weights_scale = 128;
    if (hp.n_layer != 49 || hp.n_embd != 2048 || hp.n_head != 32 || hp.n_head_kv != 4 || hp.head_dim != 128 ||
        hp.n_expert != 256 || hp.n_expert_used != 6 || hp.n_ff != 6144 || hp.n_ff_exp != 512 || hp.n_ff_shexp != 1024 ||
        hp.n_vocab != 150144)
        throw std::runtime_error("EXL3: incompatible Flint architecture");
    for (auto &e : c.get("eos_token_id").get<picojson::array>())
        eog_ids.push_back((int)e.get<double>());
    swa.assign(hp.n_layer, true);
    for (auto &e : c.get("no_rope_layer_indices").get<picojson::array>())
        swa.at((int)e.get<double>()) = false;
    auto index = json(directory + "/model.safetensors.index.json");
    std::set<std::string> names;
    for (auto &e : index.get("weight_map").get<picojson::object>())
        names.insert(e.second.get<std::string>());
    for (auto &name : names) {
        if (std::filesystem::path(name).filename() != std::filesystem::path(name))
            throw std::runtime_error("shard outside model directory");
        auto f = std::make_unique<Mapping>(directory + "/" + name);
        uint64_t len;
        std::memcpy(&len, f->data, 8);
        if (len > f->size - 8 || len > (64u << 20))
            throw std::runtime_error("invalid safetensors header");
        picojson::value h;
        std::string err = picojson::parse(h, std::string((const char *)f->data + 8, (size_t)len));
        if (!err.empty() || !h.is<picojson::object>())
            throw std::runtime_error("invalid safetensors header: " + err);
        for (auto &e : h.get<picojson::object>()) {
            if (e.first == "__metadata__")
                continue;
            Exl3Tensor t;
            t.dtype = e.second.get("dtype").get<std::string>();
            auto &offsets = e.second.get("data_offsets").get<picojson::array>();
            if (offsets.size() != 2)
                throw std::runtime_error("invalid safetensors offsets");
            size_t a = (size_t)offsets[0].get<double>(), b = (size_t)offsets[1].get<double>();
            if (a > b || b > f->size - 8 - len)
                throw std::runtime_error("tensor outside file: " + e.first);
            size_t count = 1;
            for (auto &d : e.second.get("shape").get<picojson::array>()) {
                int64_t v = (int64_t)d.get<double>();
                if (v < 0 || (v && count > std::numeric_limits<size_t>::max() / (size_t)v))
                    throw std::runtime_error("invalid safetensors shape");
                t.shape.push_back(v);
                count *= (size_t)v;
            }
            size_t unit = t.dtype == "BF16" || t.dtype == "F16" || t.dtype == "I16" ? 2
                          : t.dtype == "I32" || t.dtype == "F32"                    ? 4
                                                                                    : 0;
            if (!unit || count > (b - a) / unit || count * unit != b - a)
                throw std::runtime_error("incompatible safetensors dtype/size: " + e.first);
            t.data = f->data + 8 + len + a;
            t.bytes = b - a;
            if (!tensors_.emplace(e.first, std::move(t)).second)
                throw std::runtime_error("duplicate tensor: " + e.first);
        }
        files_.push_back(std::move(f));
    }
    for (auto &e : index.get("weight_map").get<picojson::object>())
        tensor(e.first);
}
const Exl3Tensor &Exl3Model::tensor(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error("missing EXL3 tensor: " + name);
    return it->second;
}
} // namespace fns
