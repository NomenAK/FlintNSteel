// tools/fns_kld/kld_score.h - KL base reader and statistics; no engine dependency.
// Arithmetic transcribed from llama.cpp, tools/perplexity/perplexity.cpp:191-239,1750-1758,
// revision 19ab5a5e65736a35437e6b1c0ccd6562fed01b1b (MIT; see THIRD_PARTY_NOTICES.md).
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fns_kld {

// Native little-endian llama-perplexity --kl-divergence-base format, as written on Windows x64.
// Layout: 20-byte header, all token ids, then one probability block per chunk. Only the current chunk is kept in memory.
class Base {
public:
    explicit Base(const std::string & path) : in_(path, std::ios::binary) {
        if (!in_) throw std::runtime_error("cannot open KL base: " + path);
        char magic[8];
        read_exact(magic, sizeof(magic));
        if (std::memcmp(magic, "_logits_", 8)) throw std::runtime_error("invalid KL base signature");
        uint32_t ctx;
        read_exact(&ctx, sizeof(ctx));
        read_exact(&n_vocab_, sizeof(n_vocab_));
        read_exact(&n_chunk_, sizeof(n_chunk_));
        if (ctx < 4 || ctx > uint32_t(std::numeric_limits<int>::max()) || n_vocab_ <= 0 ||
            n_vocab_ > std::numeric_limits<int>::max() - 5 || n_chunk_ <= 0) {
            throw std::runtime_error("invalid KL base dimensions");
        }
        n_ctx_ = int(ctx);
        nv_ = size_t(2) * ((size_t(n_vocab_) + 1) / 2) + 4;
        const uint64_t token_bytes = uint64_t(n_ctx_) * n_chunk_ * sizeof(int32_t);
        const uint64_t chunk_bytes = uint64_t(rows()) * nv_ * sizeof(uint16_t);
        const uint64_t max_size = uint64_t(std::numeric_limits<std::streamoff>::max());
        if (token_bytes > max_size - 20 || chunk_bytes > (max_size - 20 - token_bytes) / n_chunk_) {
            throw std::runtime_error("KL base dimensions too large");
        }
        const uint64_t expected = 20 + token_bytes + chunk_bytes * n_chunk_;
        in_.seekg(0, std::ios::end);
        const auto size = in_.tellg();
        if (size < 0 || uint64_t(size) < expected) throw std::runtime_error("truncated KL base");
        in_.seekg(20, std::ios::beg);
        tokens_.resize(size_t(n_ctx_) * n_chunk_);
        read_exact(tokens_.data(), size_t(token_bytes));
        for (int32_t tok : tokens_) {
            if (tok < 0 || tok >= n_vocab_) throw std::runtime_error("token id out of vocabulary range in KL base");
        }
        probs_.resize(size_t(rows()) * nv_);
    }

    int n_ctx() const { return n_ctx_; }
    int n_vocab() const { return n_vocab_; }
    int n_chunk() const { return n_chunk_; }
    int first() const { return n_ctx_ / 2; }
    int rows() const { return n_ctx_ - 1 - first(); }
    size_t stride() const { return nv_; }
    const int32_t * tokens(int chunk) const { return tokens_.data() + size_t(chunk) * n_ctx_; }
    const uint16_t * probabilities() const { return probs_.data(); }

    void read_chunk() {
        if (next_chunk_ >= n_chunk_) throw std::runtime_error("no more chunks in KL base");
        read_exact(probs_.data(), probs_.size() * sizeof(uint16_t));
        ++next_chunk_;
    }

private:
    void read_exact(void * dst, size_t bytes) {
        if (!in_.read(static_cast<char *>(dst), std::streamsize(bytes))) {
            throw std::runtime_error("incomplete read from KL base");
        }
    }
    std::ifstream in_;
    int n_ctx_ = 0, n_vocab_ = 0, n_chunk_ = 0, next_chunk_ = 0;
    size_t nv_ = 0;
    std::vector<int32_t> tokens_;
    std::vector<uint16_t> probs_;
};

struct Totals {
    double sum_kld = 0, sum_kld2 = 0;
    double sum_nll = 0, sum_nll2 = 0;
    double sum_nll_base = 0, sum_nll_base2 = 0;
    uint64_t count = 0, n_same_top = 0;

    void add(const Totals & b) {
        sum_kld += b.sum_kld;
        sum_kld2 += b.sum_kld2;
        sum_nll += b.sum_nll;
        sum_nll2 += b.sum_nll2;
        sum_nll_base += b.sum_nll_base;
        sum_nll_base2 += b.sum_nll_base2;
        count += b.count;
        n_same_top += b.n_same_top;
    }
};

inline std::pair<double, double> mean_and_uncertainty(double sum, double sum2, uint64_t count) {
    if (count < 1) return {0., 0.};
    const double mean = sum / count;
    double uncertainty = sum2 / count - mean * mean;
    uncertainty = uncertainty > 0 && count > 10 ? std::sqrt(uncertainty / (count - 1)) : 0.;
    return {mean, uncertainty};
}

inline void score_row(int n_vocab, const float * logits, const uint16_t * base, int tok, Totals & out) {
    float max_logit = logits[0];
    int imax = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            imax = i;
        }
    }
    double sum_exp = 0.;
    for (int i = 0; i < n_vocab; ++i) sum_exp += expf(logits[i] - max_logit);
    const float log_sum_exp = float(std::log(sum_exp));
    // memcpy avoids type-punning the float scale and offset stored at the start of each uint16_t row.
    float scale, min_log_prob;
    std::memcpy(&scale, base, sizeof(float));
    std::memcpy(&min_log_prob, base + 2, sizeof(float));
    base += 4;
    const float nll = max_logit + log_sum_exp - logits[tok];
    out.sum_nll += nll;
    out.sum_nll2 += nll * nll;
    const float nll_base = -(scale * base[tok] + min_log_prob);
    out.sum_nll_base += nll_base;
    out.sum_nll_base2 += nll_base * nll_base;
    max_logit += log_sum_exp;
    double sum = 0.;
    int imax_base = -1;
    float p_log_base_max = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float p_log_base = scale * base[i] + min_log_prob;
        if (i == 0 || p_log_base > p_log_base_max) {
            p_log_base_max = p_log_base;
            imax_base = i;
        }
        if (p_log_base > -16.f) {
            const float p_base = expf(p_log_base);
            sum += p_base * (p_log_base - logits[i] + max_logit);
        }
    }
    out.sum_kld += sum;
    out.sum_kld2 += sum * sum;
    ++out.count;
    if (imax == imax_base) ++out.n_same_top;
}

// hardware_concurrency - 1 worker threads plus the calling thread, as in llama-perplexity.
// Reduction runs in fixed order; each thread scores whole rows without locks.
class Scorer {
public:
    explicit Scorer(int rows) {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const size_t count = std::min(size_t(hw), size_t(rows));
        workers_.reserve(count - 1);
        partial_.resize(count);
    }
    size_t threads() const { return partial_.size(); }

    void score(const Base & base, const float * scored_logits, const int32_t * scored_tokens, Totals & total) {
        auto compute = [&](size_t worker) {
            Totals local;
            const int begin = int(size_t(base.rows()) * worker / partial_.size());
            const int end = int(size_t(base.rows()) * (worker + 1) / partial_.size());
            for (int row = begin; row < end; ++row) {
                score_row(base.n_vocab(), scored_logits + size_t(row) * base.n_vocab(),
                          base.probabilities() + size_t(row) * base.stride(), scored_tokens[row + 1], local);
            }
            partial_[worker] = local;
        };
        try {
            for (size_t i = 1; i < partial_.size(); ++i) workers_.emplace_back(compute, i);
        } catch (...) {
            for (auto & worker : workers_) worker.join();
            workers_.clear();
            throw;
        }
        compute(0);
        for (auto & worker : workers_) worker.join();
        workers_.clear();
        for (const auto & local : partial_) total.add(local);
    }

private:
    std::vector<std::thread> workers_;
    std::vector<Totals> partial_;
};

}  // namespace fns_kld
