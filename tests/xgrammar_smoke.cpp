// tests/xgrammar_smoke.cpp - xgrammar v0.2.1 C++ library, built with MSVC 2022 without TVM-FFI or Python, runs a
// matcher over the Flint vocabulary.
//
//   fns-xgrammar-smoke --tokenizer tokenizer.json --vocab vocab.json
//
// Build: CMake target fns-xgrammar-smoke (tests/xgrammar_smoke.cpp in CMakeLists.txt), after tools/xgrammar/build.bat has built xgrammar.lib; inputs: tools/xgrammar/export_vocab.py.
//
// vocab.json: JSON list of the 150 144 token strings (id order, from tokenizer.json). Compiles a tool-call schema,
// has the matcher accept the tokens of a valid JSON object while checking at each step that the next token is allowed
// by the mask, and measures the mean time of FillNextTokenBitmask.
#include <xgrammar/xgrammar.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "picojson.h"

static std::string slurp(const char * path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static int run(int argc, char ** argv);

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "exception : %s\n", e.what());
        return 1;
    }
}

static int run(int argc, char ** argv) {
    const char * tok_path = nullptr, * vocab_path = nullptr;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "--tokenizer") tok_path = argv[i + 1];
        else if (std::string(argv[i]) == "--vocab") vocab_path = argv[i + 1];
    }
    if (!tok_path || !vocab_path) {
        std::fprintf(stderr, "usage: fns-xgrammar-smoke --tokenizer tokenizer.json --vocab vocab.json\n");
        return 2;
    }
    picojson::value v;
    const std::string err = picojson::parse(v, slurp(vocab_path));
    if (!err.empty()) {
        std::fprintf(stderr, "vocab.json : %s\n", err.c_str());
        return 2;
    }
    std::vector<std::string> vocab;
    for (const auto & s : v.get<picojson::array>()) vocab.push_back(s.get<std::string>());

    std::printf("vocabulary read: %zu tokens\n", vocab.size());
    std::fflush(stdout);
    const auto t0 = std::chrono::steady_clock::now();
    const std::string metadata = xgrammar::TokenizerInfo::DetectMetadataFromHF(slurp(tok_path));
    picojson::value md;
    picojson::parse(md, metadata);  // {"vocab_type": int, "add_prefix_space": bool}
    const auto vocab_type = (xgrammar::VocabType) (int) md.get("vocab_type").get<double>();
    const bool prefix_space = md.get("add_prefix_space").get<bool>();
    xgrammar::TokenizerInfo info(vocab, vocab_type, (int) vocab.size(), std::vector<int32_t>{150001}, prefix_space);
    xgrammar::GrammarCompiler compiler(info, 8, false);
    // Tool-call schema (Flint call shape, with no ordering constraints beyond those of xgrammar).
    const std::string schema = R"({"type":"object","properties":{"name":{"type":"string"},)"
                               R"("arguments":{"type":"object","properties":{"path":{"type":"string"},)"
                               R"("line":{"type":"integer"}},"required":["path","line"]}},"required":["name","arguments"]})";
    // Without free whitespace, xgrammar expects json.dumps format: ", " between members and ": " after keys.
    xgrammar::CompiledGrammar cg = compiler.CompileJSONSchema(schema, false);
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("metadata %s ; vocabulary %d ; compilation %.0f ms\n", metadata.c_str(), info.GetVocabSize(),
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    // One valid tool call, tokenized character by character through the single-character vocabulary tokens
    // (byte-level: the space and printable ASCII characters each have one token).
    const std::string text = R"({"name": "read_file", "arguments": {"path": "src/engine.cu", "line": 42}})";
    std::vector<int32_t> char_tok(256, -1);
    const std::vector<std::string> & dec = info.GetDecodedVocab();
    for (int i = 0; i < (int) dec.size(); ++i)
        if (dec[i].size() == 1 && char_tok[(unsigned char) dec[i][0]] < 0) char_tok[(unsigned char) dec[i][0]] = i;
    xgrammar::GrammarMatcher m(cg);
    const int words = xgrammar::GetBitmaskSize(info.GetVocabSize());
    std::vector<int32_t> mask(words);
    DLTensor t{};
    int64_t shape[1] = {words};
    t.data = mask.data();
    t.device = DLDevice{kDLCPU, 0};
    t.ndim = 1;
    t.dtype = xgrammar::GetBitmaskDLType();
    t.shape = shape;
    double fill_us = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        const int32_t id = char_tok[(unsigned char) text[i]];
        const auto a = std::chrono::steady_clock::now();
        m.FillNextTokenBitmask(&t);
        fill_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count();
        const bool allowed = (mask[id / 32] >> (id % 32)) & 1;
        if (id < 0 || !allowed || !m.AcceptToken(id)) {
            std::fprintf(stderr, "character %zu ('%c', token %d) refused (mask: %d)\n", i, text[i], id, (int) allowed);
            return 1;
        }
    }
    m.FillNextTokenBitmask(&t);
    const bool stop_ok = (mask[150001 / 32] >> (150001 % 32)) & 1;
    const bool brace_ok = (mask[char_tok['}'] / 32] >> (char_tok['}'] % 32)) & 1;
    std::printf("%zu tokens accepted ; FillNextTokenBitmask %.1f us on average ; after the object: end %s, '}' %s\n",
                text.size(), fill_us / text.size(), stop_ok ? "allowed" : "refused", brace_ok ? "allowed" : "refused");
    return stop_ok && !brace_ok ? 0 : 1;
}
