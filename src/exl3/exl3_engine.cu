/*
 * SwiGLU dense transcribed from exllamav3 v1.6.2 (4c842731), Flint patch 3b013de9,
 * exllamav3/exllamav3_ext/activation_kernels.cuh; F32 storage, F16 operations unchanged.
 * RMSNorm and RoPE follow norm.cu and rope.cu of the same exllamav3_ext;
 * reduction order and F16 rounding adapted to the engine's F32 buffers.
 * Decode attention adapted from modules/attention_fn/triton_paged.py:890-1109 of the same project:
 * F32 softmax, P rounded to F16 before P*V, sinks added after the products, F16 output.
 * Copyright (c) 2025 Turboderp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "exl3_engine.h"
#include "../kernels.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>
#define CK(x)                                                                                                          \
    do {                                                                                                               \
        auto e = (x);                                                                                                  \
        if (e != cudaSuccess)                                                                                          \
            throw std::runtime_error(std::string(#x) + " : " + cudaGetErrorString(e));                                 \
    } while (0)
#define BL(x)                                                                                                          \
    do {                                                                                                               \
        if ((x) != CUBLAS_STATUS_SUCCESS)                                                                              \
            throw std::runtime_error("cuBLAS EXL3");                                                                   \
    } while (0)
namespace fns {
namespace {
__global__ void cvt_half(const float *x, half *y, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n)
        y[i] = __float2half_rn(x[i]);
}
__global__ void cvt_float(const half *x, float *y, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n)
        y[i] = __half2float(x[i]);
}
__global__ void embed(const __nv_bfloat16 *table, const int32_t *tokens, float *x, int rows, int stride) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < rows * 2048)
        x[i] = __bfloat162float(table[(size_t)tokens[(i / 2048) * stride] * 2048 + i % 2048]);
}
__global__ void pack_qkv(const float *q, const float *k, const float *v, float *out, int rows) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < rows * 5120) {
        int r = i / 5120, c = i % 5120;
        out[i] = c < 4096 ? q[r * 4096 + c] : c < 4608 ? k[r * 512 + c - 4096] : v[r * 512 + c - 4608];
    }
}
__global__ void routing_pack(const int32_t *ids, const float *weights, int64_t *sel, half *rw, int n) {
    int i = threadIdx.x;
    if (i < n) {
        sel[i] = ids[i];
        rw[i] = __float2half_rn(weights[i] * 128.f);
    }
}
// exllamav3 activation_kernels.cuh: each dense operation is rounded to F16.
__global__ void dense_swiglu(const float *g, const float *u, float *a, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        half v = __float2half_rn(g[i]);
        half s = __hmul(v, hrcp(__hadd(__float2half_rn(1.f), hexp(__hneg(v)))));
        a[i] = __half2float(__hmul(s, __float2half_rn(u[i])));
    }
}
// norm.cu: groups of four contiguous values, FMA then XOR reduction, weight applied before the RMS scale.
__global__ void exl_norm(const float *x, const float *w, float *y) {
    __shared__ float sums[16];
    int t = threadIdx.x, lane = t % 32, warp = t / 32, row = blockIdx.x;
    float4 v = ((const float4 *)(x + row * 2048))[t];
    float s = fmaf(v.x, v.x, 0.f);
    s = fmaf(v.y, v.y, s);
    s = fmaf(v.z, v.z, s);
    s = fmaf(v.w, v.w, s);
    for (int o = 16; o; o /= 2)
        s += __shfl_xor_sync(0xffffffff, s, o);
    if (!lane)
        sums[warp] = s;
    __syncthreads();
    s = lane < 16 ? sums[lane] : 0.f;
    for (int o = 16; o; o /= 2)
        s += __shfl_xor_sync(0xffffffff, s, o);
    float r = rsqrtf(s / 2048.f + 1e-6f);
    float4 a = ((const float4 *)w)[t];
    v.x = __half2float(__float2half_rn((v.x * a.x) * r));
    v.y = __half2float(__float2half_rn((v.y * a.y) * r));
    v.z = __half2float(__float2half_rn((v.z * a.z) * r));
    v.w = __half2float(__float2half_rn((v.w * a.w) * r));
    ((float4 *)(y + row * 2048))[t] = v;
}
// rope.cu / norm.cu: norm output rounded to F16 before RoPE, then F16 storage.
__global__ void exl_qk(float *qkv, float *qout, const float *qn, const float *kn, half *kc, half *vc, const int32_t *io,
                       int p0, int slots, bool rope) {
    __shared__ float sums[2];
    __shared__ half values[128];
    int row = blockIdx.y, head = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
    int base = head < 32 ? head * 128 : 4096 + (head - 32) * 128;
    float *src = qkv + row * 5120 + base;
    const float *w = head < 32 ? qn : kn;
    float s = 0.f;
    if (rope) {
        float a = src[2 * t], b = src[2 * t + 1];
        s = a * a + b * b;
    } else {
        float4 v = ((float4 *)src)[t % 32];
        if (t < 32) {
            s = fmaf(v.x, v.x, 0.f);
            s = fmaf(v.y, v.y, s);
            s = fmaf(v.z, v.z, s);
            s = fmaf(v.w, v.w, s);
        }
    }
    if (rope) {
        for (int o = 16; o; o /= 2)
            s += __shfl_down_sync(0xffffffff, s, o);
    } else {
        for (int o = 16; o; o /= 2)
            s += __shfl_xor_sync(0xffffffff, s, o);
    }
    if (!lane)
        sums[warp] = s;
    __syncthreads();
    s = rope ? sums[0] + sums[1] : sums[0];
    float r = rsqrtf(s / 128.f + 1e-6f);
    for (int i = t; i < 128; i += 64)
        values[i] = __float2half_rn(rope ? (src[i] * r) * w[i] : (src[i] * w[i]) * r);
    __syncthreads();
    int pos = io ? io[1] : p0 + row;
    float a = __half2float(values[t]), b = __half2float(values[t + 64]);
    if (rope) {
        float fr = 1.f / powf(10000.f, (float)(2 * t) / 128.f), theta = fr * (float)pos;
        float turns = rintf(theta * 0.15915494309189535f);
        float rr = fmaf(-turns, 6.28318548202514648f, theta);
        rr = fmaf(-turns, -1.74845553e-7f, rr);
        float sn, cs;
        __sincosf(rr, &sn, &cs);
        float aa = a * cs - b * sn;
        b = b * cs + a * sn;
        a = aa;
    }
    half ha = __float2half_rn(a), hb = __float2half_rn(b);
    if (head < 32) {
        qout[row * 4096 + head * 128 + t] = __half2float(ha);
        qout[row * 4096 + head * 128 + t + 64] = __half2float(hb);
    } else {
        size_t off = (size_t)(pos % slots) * 512 + (head - 32) * 128 + t;
        kc[off] = ha;
        kc[off + 64] = hb;
        vc[off] = __float2half_rn(qkv[row * 5120 + 4608 + (head - 32) * 128 + t]);
        vc[off + 64] = __float2half_rn(qkv[row * 5120 + 4608 + (head - 32) * 128 + t + 64]);
    }
}
// exllamav3 v1.6.2 triton_paged.py:992-1037: QK scale follows the dot, P is cast to
// F16 before P*V, the denominator stays F32, and the final attention activation is F16.
// EXL3-only; the GGUF attention deliberately retains its own arithmetic and kernels.
__global__ void exl_attn_part(const float *q, const half *kc, const half *vc, float *part,
                              const int32_t *position, int slots, int window, int chunks) {
    __shared__ float scores[64], probabilities[64], reduction[4];
    int t = threadIdx.x, lane = t % 32, warp = t / 32, h = blockIdx.y;
    int pos = *position, first = window ? max(0, pos - window + 1) : 0;
    int begin = (first / 64 + blockIdx.x) * 64;
    if (begin > pos)
        return;
    for (int j = warp; j < 64; j += 4) {
        int p = begin + j;
        float dot = 0.f;
        if (p >= first && p <= pos) {
            size_t base = (size_t)(p % slots) * 512 + (h / 8) * 128;
            for (int c = 0; c < 4; ++c)
                dot = fmaf(q[h * 128 + lane + c * 32], __half2float(kc[base + lane + c * 32]), dot);
        }
        for (int offset = 16; offset; offset /= 2)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (!lane)
            scores[j] = p >= first && p <= pos ? dot * 0.088388347648f : -INFINITY;
    }
    __syncthreads();
    float maximum = t < 64 ? scores[t] : -INFINITY;
    for (int offset = 16; offset; offset /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    if (!lane)
        reduction[warp] = maximum;
    __syncthreads();
    maximum = fmaxf(fmaxf(reduction[0], reduction[1]), fmaxf(reduction[2], reduction[3]));
    float probability = t < 64 ? expf(scores[t] - maximum) : 0.f;
    if (t < 64)
        probabilities[t] = __half2float(__float2half_rn(probability));
    for (int offset = 16; offset; offset /= 2)
        probability += __shfl_xor_sync(0xffffffff, probability, offset);
    if (!lane)
        reduction[warp] = probability;
    __syncthreads();
    float sum = (reduction[0] + reduction[1]) + (reduction[2] + reduction[3]);
    float output = 0.f;
    for (int j = 0; j < 64 && begin + j <= pos; ++j)
        if (begin + j >= first) {
            size_t base = (size_t)((begin + j) % slots) * 512 + (h / 8) * 128;
            output = fmaf(probabilities[j], __half2float(vc[base + t]), output);
        }
    size_t index = (size_t)h * chunks + blockIdx.x;
    part[index * 128 + t] = output;
    if (!t) {
        float2 *ms = (float2 *)(part + (size_t)32 * chunks * 128);
        ms[index] = make_float2(maximum, sum);
    }
}
__global__ void exl_attn_reduce(const float *part, const float *sinks, float *out,
                                const int32_t *position, int window, int chunks) {
    int h = blockIdx.x, d = threadIdx.x, pos = *position;
    int first = window ? max(0, pos - window + 1) : 0;
    int used = pos / 64 - first / 64 + 1;
    const float2 *ms = (const float2 *)(part + (size_t)32 * chunks * 128) + h * chunks;
    float maximum = sinks[h];
    for (int c = 0; c < used; ++c)
        maximum = fmaxf(maximum, ms[c].x);
    float sum = expf(sinks[h] - maximum), output = 0.f;
    for (int c = 0; c < used; ++c) {
        float alpha = expf(ms[c].x - maximum);
        sum += ms[c].y * alpha;
        output += part[((size_t)h * chunks + c) * 128 + d] * alpha;
    }
    out[h * 128 + d] = __half2float(__float2half_rn(output / sum));
}
} // namespace
void Exl3Engine::decode_attention(const float *q, const half *k, const half *v, float *part,
                                  const float *sinks, float *out, const int32_t *position,
                                  int slots, int window, int chunks, cudaStream_t stream) {
    exl_attn_part<<<dim3(chunks, 32), 128, 0, stream>>>(q, k, v, part, position, slots, window, chunks);
    exl_attn_reduce<<<32, 128, 0, stream>>>(part, sinks, out, position, window, chunks);
}
void *Exl3Engine::allocate(size_t n) {
    void *p;
    CK(cudaMalloc(&p, n));
    allocs_.push_back(p);
    return p;
}
void *Exl3Engine::upload(const Exl3Tensor &t) {
    void *p = allocate(t.bytes);
    CK(cudaMemcpyAsync(p, t.data, t.bytes, cudaMemcpyHostToDevice, st_));
    CK(cudaStreamSynchronize(st_));
    weight_bytes_ += t.bytes;
    return p;
}
float *Exl3Engine::norm(const std::string &name) {
    const auto &t = m_.tensor(name);
    if (t.dtype == "F32")
        return (float *)upload(t);
    if (t.dtype != "BF16" && t.dtype != "F16")
        throw std::runtime_error("float norm required: " + name);
    std::vector<float> f(t.bytes / 2);
    const auto *b = (const uint16_t *)t.data;
    for (size_t i = 0; i < f.size(); ++i) {
        if (t.dtype == "F16")
            f[i] = __half2float(((const half *)t.data)[i]);
        else {
            uint32_t v = (uint32_t)b[i] << 16;
            std::memcpy(&f[i], &v, 4);
        }
    }
    auto *p = (float *)allocate(f.size() * 4);
    CK(cudaMemcpyAsync(p, f.data(), f.size() * 4, cudaMemcpyHostToDevice, st_));
    CK(cudaStreamSynchronize(st_));
    weight_bytes_ += f.size() * 4;
    return p;
}
exl3::Linear Exl3Engine::linear(const std::string &key) {
    const auto &t = m_.tensor(key + ".trellis");
    const auto &u = m_.tensor(key + ".suh");
    const auto &v = m_.tensor(key + ".svh");
    const auto &marker = m_.tensor(key + ".mul1");
    uint32_t codebook = 0;
    if (marker.bytes == 4)
        std::memcpy(&codebook, marker.data, 4);
    if (t.dtype != "I16" || t.shape.size() != 3 || u.dtype != "F16" || v.dtype != "F16" || marker.dtype != "I32" ||
        codebook != 0x83DCD12D || (t.shape[2] != 64 && t.shape[2] != 80) ||
        u.shape != std::vector<int64_t>{t.shape[0] * 16} || v.shape != std::vector<int64_t>{t.shape[1] * 16})
        throw std::runtime_error("incompatible EXL3 weight: " + key);
    return {(uint16_t *)upload(t), (half *)upload(u),    (half *)upload(v),
            (int)t.shape[0] * 16,  (int)t.shape[1] * 16, (int)t.shape[2] / 16};
}
Exl3Engine::Exl3Engine(const Exl3Model &model, const EngineOptions &opt) : m_(model), opt_(opt) {
    std::fprintf(
        stderr,
        "EXL3 experimental: kernel parity validated, model parity not validated; GGUF remains recommended.\n");
    if (opt_.n_ctx <= 0 || opt_.ring < 513)
        throw std::runtime_error("EXL3: invalid context/ring size");
    opt_.ubatch = std::max(1, std::min(opt_.ubatch > 0 ? opt_.ubatch : 8, 8));
    opt_.ring = (std::max(opt_.ring, 513 + opt_.ubatch - 1) + 255) / 256 * 256;
    CK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    BL(cublasCreate(&blas_));
    BL(cublasSetStream(blas_, st_));
    exl3::initialize();
    auto &emb = m_.tensor("model.embed_tokens.weight");
    if (emb.dtype != "BF16")
        throw std::runtime_error("BF16 embedding required");
    CK(cudaHostAlloc(&embedding_host_, emb.bytes, cudaHostAllocMapped));
    std::memcpy(embedding_host_, emb.data, emb.bytes);
    CK(cudaHostGetDevicePointer((void **)&embedding_, embedding_host_, 0));
    norm_ = norm("model.norm.weight");
    head_ = linear("lm_head");
    layers_.resize(49);
    placement_.resident.assign(49 * 256, 0);
    for (int il = 0; il < 49; ++il) {
        auto &l = layers_[il];
        std::string key = "model.layers." + std::to_string(il);
        l.an = norm(key + ".input_layernorm.weight");
        l.fn = norm(key + ".post_attention_layernorm.weight");
        l.qn = norm(key + ".self_attn.q_norm.weight");
        l.kn = norm(key + ".self_attn.k_norm.weight");
        l.sinks = norm(key + ".self_attn.softmax_offset");
        l.q = linear(key + ".self_attn.q_proj");
        l.k = linear(key + ".self_attn.k_proj");
        l.v = linear(key + ".self_attn.v_proj");
        l.o = linear(key + ".self_attn.o_proj");
        if (il < 2) {
            l.g = linear(key + ".mlp.gate_proj");
            l.u = linear(key + ".mlp.up_proj");
            l.d = linear(key + ".mlp.down_proj");
        } else {
            l.router.weight = (half *)upload(m_.tensor(key + ".mlp.gate.weight"));
            l.router.hi = (signed char *)allocate(2 * 256 * 2048);
            l.router.scales = (float *)allocate(256 * 4);
            exl3::prepare_router(l.router, st_);
            l.sg = linear(key + ".mlp.shared_expert.gate_proj");
            l.su = linear(key + ".mlp.shared_expert.up_proj");
            l.sd = linear(key + ".mlp.shared_expert.down_proj");
            const exl3::Linear *shw[] = {&l.sg, &l.su, &l.sd};
            const int64_t *sht[9];
            for (int j = 0; j < 3; ++j) {
                int64_t host[] = {(int64_t)shw[j]->trellis, (int64_t)shw[j]->suh, (int64_t)shw[j]->svh};
                for (int z = 0; z < 3; ++z) {
                    sht[j * 3 + z] = (int64_t *)allocate(8);
                    CK(cudaMemcpy((void *)sht[j * 3 + z], host + z, 8, cudaMemcpyHostToDevice));
                }
            }
            auto &sh = l.shared;
            sh.g_trellis = sht[0];
            sh.g_suh = sht[1];
            sh.g_svh = sht[2];
            sh.u_trellis = sht[3];
            sh.u_suh = sht[4];
            sh.u_svh = sht[5];
            sh.d_trellis = sht[6];
            sh.d_suh = sht[7];
            sh.d_svh = sht[8];
            size_t before = weight_bytes_;
            std::vector<int64_t> tables[9];
            const char *projections[] = {"gate", "up", "down"};
            for (int p = 0; p < 3; ++p)
                for (int e = 0; e < 256; ++e) {
                    auto w = linear(key + ".mlp.experts." + std::to_string(e) + "." + projections[p] + "_proj");
                    tables[p * 3].push_back((int64_t)w.trellis);
                    tables[p * 3 + 1].push_back((int64_t)w.suh);
                    tables[p * 3 + 2].push_back((int64_t)w.svh);
                }
            const int64_t *t[9];
            for (int j = 0; j < 9; ++j) {
                t[j] = (int64_t *)allocate(256 * 8);
                CK(cudaMemcpy((void *)t[j], tables[j].data(), 256 * 8, cudaMemcpyHostToDevice));
            }
            auto &p = l.moe;
            p.g_trellis = t[0];
            p.g_suh = t[1];
            p.g_svh = t[2];
            p.u_trellis = t[3];
            p.u_suh = t[4];
            p.u_svh = t[5];
            p.d_trellis = t[6];
            p.d_suh = t[7];
            p.d_svh = t[8];
            placement_.vram_bytes += weight_bytes_ - before;
            placement_.vram_experts += 256;
            std::fill(placement_.resident.begin() + il * 256, placement_.resident.begin() + (il + 1) * 256, 1);
        }
        l.slots = m_.swa[il] ? opt_.ring : opt_.n_ctx;
        size_t kv = (size_t)((l.slots + 255) / 256 * 256) * 512 * 2;
        l.kc = allocate(kv);
        l.vc = allocate(kv);
        CK(cudaMemset(l.kc, 0, kv));
        CK(cudaMemset(l.vc, 0, kv));
    }
    auto f = [&](int n) { return (float *)allocate((size_t)n * 4); };
    auto h = [&](int n) { return (half *)allocate((size_t)n * 2); };
    x_ = f(8 * 2048);
    h_ = f(8 * 2048);
    qkv_ = f(8 * 5120);
    q_ = f(8 * 4096);
    att_ = f(8 * 4096);
    out_ = f(8 * 2048);
    gate_ = f(8 * 6144);
    up_ = f(8 * 6144);
    act_ = f(8 * 6144);
    shared_ = f(8 * 2048);
    routed_ = f(8 * 2048);
    router_ = f(8 * 256);
    rw_f_ = f(8 * 6);
    xh_ = h(8 * 6144);
    had_ = h(8 * 6144);
    yh_ = h(8 * 150144);
    rw_ = h(48);
    logits_ = f(150144);
    scores_ = f(32 * ((std::max(opt_.n_ctx, 576) + 63) / 64) * 130);
    mask_cols_ = std::max((opt_.n_ctx + 255) / 256 * 256, opt_.ring);
    mask_ = h(mask_cols_ * 8);
    io_ = (int32_t *)allocate(4 * 4);
    tokens_ = (int32_t *)allocate(8 * 4);
    ids_ = (int32_t *)allocate(48 * 4);
    selected_ = (int64_t *)allocate(48 * 8);
    locks_ = (int *)allocate(8 << 20);
    CK(cudaMemset(locks_, 0, 8 << 20));
    CK(cudaMallocHost(&host_io_, 4 * 4));
    auto &s = scratch_;
    s.H = s.Hi = s.Ho = s.H_out = 2048;
    s.I = 512;
    s.min_expert = s.max_expert = -1;
    s.n_local = 256;
    s.gated = true;
    s.gu_f32 = false;
    s.slots_max = 48;
    s.rows_max = 8;
    s.x_stride = 2048;
    s.out_stride = 2048;
    s.had_g = h(48 * 2048);
    s.had_u = h(48 * 2048);
    s.gu_g = h(48 * 512);
    s.gu_u = h(48 * 512);
    s.act_out = h(48 * 512);
    s.d_out = f(48 * 2048);
    s.ctr_a_len = 48 * 4;
    s.ctr_b_len = 8 * 16;
    s.ctr_a = (int *)allocate(exl3_moe_coop_ctr_len(48, 8, 512, 2048) * 4);
    CK(cudaMemset(s.ctr_a, 0, exl3_moe_coop_ctr_len(48, 8, 512, 2048) * 4));
    s.ctr_b = s.ctr_a + s.ctr_a_len;
    s.runs = s.ctr_b + s.ctr_b_len;
    auto &sh = shared_scratch_;
    sh = s;
    sh.I = 1024;
    sh.slots_max = 8;
    sh.n_local = 1;
    sh.gu_f32 = false;
    sh.had_g = h(8 * 2048);
    sh.had_u = h(8 * 2048);
    sh.gu_g = h(8 * 1024);
    sh.gu_u = h(8 * 1024);
    sh.act_out = h(8 * 1024);
    sh.d_out = f(8 * 2048);
    sh.ctr_a_len = 8 * 8;
    sh.ctr_b_len = 8 * 16;
    sh.ctr_a = (int *)allocate(exl3_moe_coop_ctr_len(8, 8, 1024, 2048) * 4);
    CK(cudaMemset(sh.ctr_a, 0, exl3_moe_coop_ctr_len(8, 8, 1024, 2048) * 4));
    sh.ctr_b = sh.ctr_a + sh.ctr_a_len;
    sh.runs = sh.ctr_b + sh.ctr_b_len;
    shared_sel_ = (int64_t *)allocate(8 * 8);
    CK(cudaMemset(shared_sel_, 0, 8 * 8));
    shared_rw_ = h(8);
    half ones[8];
    for (auto &v : ones)
        v = __float2half_rn(1.f);
    CK(cudaMemcpy(shared_rw_, ones, sizeof(ones), cudaMemcpyHostToDevice));
    router_scratch_.hi = (signed char *)allocate(8 * 2048);
    router_scratch_.lo = (signed char *)allocate(8 * 2048);
    router_scratch_.scales = f(8 * 16);
    router_scratch_.partial = f(8 * 8 * 256);
    router_scratch_.scores = h(8 * 256);
    mask_dev_ = (uint32_t *)allocate(((150144 + 31) / 32) * 4);
    CK(cudaMallocHost(&mask_host_, ((150144 + 31) / 32) * 4));
    seen_ = f(150144);
    CK(cudaMemset(seen_, 0, 150144 * 4));
    CK(cudaEventCreateWithFlags(&pick_ev_, cudaEventDisableTiming));
    // Pointer-table uploads and initialization memsets use the default stream; complete setup
    // before the first non-blocking engine-stream forward can consume any of these buffers.
    CK(cudaStreamSynchronize(nullptr));
    CK(cudaStreamSynchronize(st_));
}
Exl3Engine::~Exl3Engine() {
    if (st_)
        cudaStreamSynchronize(st_);
    if (graph_)
        cudaGraphExecDestroy(graph_);
    if (pick_ev_)
        cudaEventDestroy(pick_ev_);
    if (blas_)
        cublasDestroy(blas_);
    for (auto p : allocs_)
        cudaFree(p);
    if (host_io_)
        cudaFreeHost(host_io_);
    if (mask_host_)
        cudaFreeHost(mask_host_);
    if (embedding_host_)
        cudaFreeHost(embedding_host_);
    if (st_)
        cudaStreamDestroy(st_);
}
void Exl3Engine::project(const exl3::Linear &w, const float *x, float *y, int rows, bool half_output) {
    cvt_half<<<(rows * w.k + 255) / 256, 256, 0, st_>>>(x, xh_, rows * w.k);
    exl3::gemv(w, xh_, rows, had_, half_output ? (void *)yh_ : (void *)y, !half_output, locks_, st_);
    if (half_output)
        cvt_float<<<(rows * w.n + 255) / 256, 256, 0, st_>>>(yh_, y, rows * w.n);
}
template <bool TraceEnabled>
void Exl3Engine::forward(int rows, int p0, bool head_all, float *host_logits, float *layer_out, Trace *trace) {
    int trace_p0 = p0;
    if constexpr (TraceEnabled) {
        if (p0 < 0) {
            CK(cudaMemcpyAsync(&trace_p0, io_ + 1, 4, cudaMemcpyDeviceToHost, st_));
            CK(cudaStreamSynchronize(st_));
        }
    }
    auto tap = [&](int layer, const char *name, void *data, int width, bool fp16 = false, int row = -1) {
        if constexpr (TraceEnabled)
            trace->visit(trace->context, row < 0 ? trace_p0 : trace_p0 + row, layer, name, data,
                         row < 0 ? rows : 1, width, fp16, st_);
    };
    bool decode = p0 < 0;
    embed<<<(rows * 2048 + 255) / 256, 256, 0, st_>>>(embedding_, decode ? io_ : tokens_, x_, rows, 1);
    tap(-1, "embedding", x_, 2048);
    for (int il = 0; il < 49; ++il) {
        auto &l = layers_[il];
        tap(il, "input", x_, 2048);
        exl_norm<<<rows, 512, 0, st_>>>(x_, l.an, h_);
        tap(il, "attn_norm", h_, 2048);
        project(l.q, h_, q_, rows, true);
        tap(il, "q_proj", q_, 4096);
        project(l.k, h_, gate_, rows, true);
        tap(il, "k_proj", gate_, 512);
        project(l.v, h_, up_, rows, true);
        tap(il, "v_proj", up_, 512);
        pack_qkv<<<(rows * 5120 + 255) / 256, 256, 0, st_>>>(q_, gate_, up_, qkv_, rows);
        exl_qk<<<dim3(36, rows), 64, 0, st_>>>(qkv_, q_, l.qn, l.kn, (half *)l.kc, (half *)l.vc, decode ? io_ : nullptr,
                                               p0, l.slots, m_.swa[il]);
        tap(il, "q_post", q_, 4096);
        if constexpr (TraceEnabled)
            for (int row = 0; row < rows; ++row) {
                size_t offset = (size_t)((trace_p0 + row) % l.slots) * 512;
                tap(il, "k_post", (half *)l.kc + offset, 512, true, row);
                tap(il, "v_post", (half *)l.vc + offset, 512, true, row);
            }
        if (decode) {
            int window = m_.swa[il] ? 513 : 0;
            int chunks = ((window ? window + 63 : opt_.n_ctx) + 63) / 64;
            decode_attention(q_, (half *)l.kc, (half *)l.vc, scores_, l.sinks, att_, io_ + 1,
                             l.slots, window, chunks, st_);
        } else {
            int nk = m_.swa[il] ? l.slots : ((p0 + rows + 255) / 256 * 256);
            k::prefill_mask(mask_, nk, p0, rows, l.slots, m_.swa[il] ? 513 : 0, st_);
            flash_attn_prefill(q_, l.kc, l.vc, mask_, l.sinks, att_, rows, nk, 32, 4, 128, 0.088388347648f, st_);
        }
        tap(il, "attention", att_, 4096);
        project(l.o, att_, out_, rows);
        tap(il, "o_proj", out_, 2048);
        k::add_inplace(x_, out_, rows * 2048, st_);
        tap(il, "attn_residual", x_, 2048);
        exl_norm<<<rows, 512, 0, st_>>>(x_, l.fn, h_);
        tap(il, "mlp_norm", h_, 2048);
        if (il < 2) {
            project(l.g, h_, gate_, rows, true);
            tap(il, "gate_proj", gate_, 6144);
            project(l.u, h_, up_, rows, true);
            tap(il, "up_proj", up_, 6144);
            dense_swiglu<<<(rows * 6144 + 255) / 256, 256, 0, st_>>>(gate_, up_, act_, rows * 6144);
            tap(il, "activation", act_, 6144);
            project(l.d, act_, out_, rows);
            tap(il, "mlp", out_, 2048);
            k::add_inplace(x_, out_, rows * 2048, st_);
        } else {
            cvt_half<<<(rows * 2048 + 255) / 256, 256, 0, st_>>>(h_, xh_, rows * 2048);
            exl3::route(l.router, xh_, rows, router_scratch_, selected_, rw_, st_);
            tap(il, "router_input", h_, 2048);
            auto sh = shared_scratch_;
            sh.g_trellis = l.shared.g_trellis;
            sh.g_suh = l.shared.g_suh;
            sh.g_svh = l.shared.g_svh;
            sh.u_trellis = l.shared.u_trellis;
            sh.u_suh = l.shared.u_suh;
            sh.u_svh = l.shared.u_svh;
            sh.d_trellis = l.shared.d_trellis;
            sh.d_suh = l.shared.d_suh;
            sh.d_svh = l.shared.d_svh;
            cvt_half<<<(rows * 2048 + 255) / 256, 256, 0, st_>>>(h_, xh_, rows * 2048);
            sh.x = xh_;
            sh.sel = shared_sel_;
            sh.rw = shared_rw_;
            sh.bsz = rows;
            sh.topk = 1;
            sh.out = shared_;
            exl3::moe(sh, st_);
            tap(il, "shared", shared_, 2048);
            auto p = scratch_;
            p.g_trellis = l.moe.g_trellis;
            p.g_suh = l.moe.g_suh;
            p.g_svh = l.moe.g_svh;
            p.u_trellis = l.moe.u_trellis;
            p.u_suh = l.moe.u_suh;
            p.u_svh = l.moe.u_svh;
            p.d_trellis = l.moe.d_trellis;
            p.d_suh = l.moe.d_suh;
            p.d_svh = l.moe.d_svh;
            cvt_half<<<(rows * 2048 + 255) / 256, 256, 0, st_>>>(h_, xh_, rows * 2048);
            p.x = xh_;
            p.sel = selected_;
            p.rw = rw_;
            p.bsz = rows;
            p.topk = 6;
            p.out = routed_;
            p.sh_out = shared_;
            exl3::moe(p, st_);
            tap(il, "mlp", routed_, 2048);
            k::add_inplace(x_, routed_, rows * 2048, st_);
        }
        tap(il, "output", x_, 2048);
        if (layer_out)
            CK(cudaMemcpyAsync(layer_out + (size_t)il * 2048, x_, 2048 * 4, cudaMemcpyDeviceToHost, st_));
    }
    for (int t = head_all ? 0 : rows - 1; t < rows; ++t) {
        exl_norm<<<1, 512, 0, st_>>>(x_ + t * 2048, norm_, h_);
        if constexpr (TraceEnabled)
            trace->visit(trace->context, trace_p0 + t, 49, "head_norm", h_, 1, 2048, false, st_);
        project(head_, h_, logits_, 1, true);
        if constexpr (TraceEnabled)
            trace->visit(trace->context, trace_p0 + t, 49, "logits", logits_, 1, 150144, false, st_);
        if (host_logits)
            CK(cudaMemcpyAsync(host_logits + (size_t)t * 150144, logits_, 150144 * 4, cudaMemcpyDeviceToHost, st_));
    }
}
template void Exl3Engine::forward<true>(int, int, bool, float *, float *, Exl3Engine::Trace *);
void Exl3Engine::run_decode(float *layers) {
    if (layers || std::getenv("FNS_NO_GRAPH")) {
        forward(1, -1, false, nullptr, layers);
        return;
    }
    if (!graph_) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
        forward(1, -1, false);
        CK(cudaStreamEndCapture(st_, &g));
        CK(cudaGraphInstantiate(&graph_, g, 0));
        CK(cudaGraphDestroy(g));
    }
    CK(cudaGraphLaunch(graph_, st_));
}
void Exl3Engine::launch(int32_t token, int pos, float *layers) {
    if (token < 0 || token >= 150144 || pos < 0 || pos >= opt_.n_ctx)
        throw std::runtime_error("token/position out of range");
    host_io_[0] = token;
    host_io_[1] = pos;
    CK(cudaMemcpyAsync(io_, host_io_, 8, cudaMemcpyHostToDevice, st_));
    run_decode(layers);
}
void Exl3Engine::step(int32_t token, int pos, float *logits, float *layers) {
    launch(token, pos, layers);
    if (logits)
        CK(cudaMemcpyAsync(logits, logits_, 150144 * 4, cudaMemcpyDeviceToHost, st_));
    else {
        k::argmax(logits_, 150144, io_ + 2, st_);
        CK(cudaMemcpyAsync(host_io_ + 2, io_ + 2, 4, cudaMemcpyDeviceToHost, st_));
    }
    CK(cudaStreamSynchronize(st_));
}
int32_t Exl3Engine::step_greedy(int32_t t, int p) {
    step(t, p, nullptr);
    return host_io_[2];
}
void Exl3Engine::step_logits(int32_t t, int p) { launch(t, p); }
void Exl3Engine::step_next(int p) {
    if (p < 0 || p >= opt_.n_ctx)
        throw std::runtime_error("position outside context");
    k::io_advance(io_, p, st_);
    run_decode();
}
int32_t Exl3Engine::pick(const uint32_t *mask, const SampleParams &sp, uint64_t c) {
    pick_async(mask, sp, c);
    return pick_wait();
}
void Exl3Engine::pick_async(const uint32_t *mask, const SampleParams &sp, uint64_t c) {
    if (mask) {
        std::memcpy(mask_host_, mask, ((150144 + 31) / 32) * 4);
        CK(cudaMemcpyAsync(mask_dev_, mask_host_, ((150144 + 31) / 32) * 4, cudaMemcpyHostToDevice, st_));
    }
    if (sp.presence_penalty != 0)
        k::presence_apply(logits_, seen_, 150144, sp.presence_penalty, st_);
    if (sp.temperature <= 0)
        k::argmax(logits_, 150144, io_ + 2, st_, mask ? mask_dev_ : nullptr);
    else
        k::sample(logits_, 150144, mask ? mask_dev_ : nullptr, sp.top_k, sp.top_p, sp.temperature, sp.seed, c, io_ + 2,
                  st_);
    if (sp.presence_penalty != 0)
        k::presence_mark(seen_, io_ + 2, st_);
    CK(cudaMemcpyAsync(host_io_ + 2, io_ + 2, 4, cudaMemcpyDeviceToHost, st_));
    CK(cudaEventRecord(pick_ev_, st_));
}
int32_t Exl3Engine::pick_wait() {
    CK(cudaEventSynchronize(pick_ev_));
    return host_io_[2];
}
void Exl3Engine::reset_presence() { CK(cudaMemsetAsync(seen_, 0, 150144 * 4, st_)); }
int32_t Exl3Engine::prefill(const int32_t *ids, int n, int p0, float *logits) {
    if (n <= 0 || p0 < 0 || p0 + n > opt_.n_ctx)
        throw std::runtime_error("prefill outside context");
    for (int i = 0; i < n; ++i)
        if (ids[i] < 0 || ids[i] >= 150144)
            throw std::runtime_error("token outside vocabulary");
    for (int i = 0; i < n; i += opt_.ubatch) {
        int rows = std::min(opt_.ubatch, n - i);
        CK(cudaMemcpyAsync(tokens_, ids + i, rows * 4, cudaMemcpyHostToDevice, st_));
        forward(rows, p0 + i, logits != nullptr, logits ? logits + (size_t)i * 150144 : nullptr);
        if (logits)
            CK(cudaStreamSynchronize(st_));
    }
    k::argmax(logits_, 150144, io_ + 2, st_);
    CK(cudaMemcpyAsync(host_io_ + 2, io_ + 2, 4, cudaMemcpyDeviceToHost, st_));
    CK(cudaStreamSynchronize(st_));
    return host_io_[2];
}
} // namespace fns
