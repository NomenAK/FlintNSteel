/*
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Origin: exllamav3 v1.6.2 (4c842731), Flint patch 3b013de9.
Path: exllamav3/exllamav3_ext/quant/exl3_moe_coop.cuh
Modifications: no arithmetic changes; Tensor interface removed.
*/
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Fused decode-shaped MoE path (bsz <= MAX_BSZN): two launches per layer run every (token, expert)
// slot through gate/up, activation and down, replacing the per-projection mgemm CUDA graphs of
// BC_BlockSparseMLP. See exl3_moe_coop_kernel.cuh / exl3_moe_coop.cu.

#define MOE_COOP_ACT_SILU 0
#define MOE_COOP_ACT_GELU 1
#define MOE_COOP_ACT_RELU2 2
#define MOE_COOP_ACT_SILU_OAI 3

#define MOE_COOP_THREADS 512
#define MOE_COOP_WNT 2                  // adjacent 16-column tiles per warp (32-column groups per block)
#define MOE_COOP_COLS (MOE_COOP_WNT * 16)

struct MoeCoopParams
{
    // Inputs
    const half* x;              // (bsz, H), row stride x_stride
    int x_stride;
    const int64_t* sel;         // (bsz, topk) global expert indices
    const half* rw;             // (bsz, topk) routing weights
    int bsz;
    int topk;
    int H;                      // x width (the experts' input width before padding)
    int Hi;                     // gate/up in_features, padded (multiple of 128)
    int I;                      // intermediate width, padded (multiple of 128)
    int Ho;                     // down out_features, padded (multiple of 128)
    int H_out;                  // output width (<= Ho)
    int min_expert;             // -1: no expert-range filtering; else sel in [min, max) is local
    int max_expert;

    // Per-expert pointer tables (int64 device arrays indexed by local expert)
    const int64_t* g_trellis;   const int64_t* g_suh;   const int64_t* g_svh;
    const int64_t* u_trellis;   const int64_t* u_suh;   const int64_t* u_svh;
    const int64_t* d_trellis;   const int64_t* d_suh;   const int64_t* d_svh;
    const int64_t* g_bias;      // nullable
    const int64_t* u_bias;
    const int64_t* d_bias;
    int act;                    // MOE_COOP_ACT_*
    float act_limit;
    bool gated;                 // false: no gate projection, activation is relu(u) * u

    // Scratch (leading dim >= bsz * topk)
    half* had_g;                // (slots, Hi) rotated gate input (bsz > 1, written by the rot kernel)
    half* had_u;                // (slots, Hi) rotated up input
    bool a_global;              // true: the GEMV reads had_g / had_u; false (bsz 1): rotated in-block
    void* gu_g;                 // (slots, I) gate GEMV output, half or float (gu_f32)
    void* gu_u;                 // (slots, I)
    bool gu_f32;
    half* act_out;              // (slots, I) activation, rotated for the down projection
    float* d_out;               // (slots, Ho) down GEMV output before the output rotation
    int* ctr_a;                 // (slots_max, I / 128) completion counters of the gate/up stage
    int* ctr_b;                 // (MAX_BSZN, Ho / 128) completion counters of the down stage
    int ctr_a_len;
    int ctr_b_len;
    int ksplit_a;               // split-k blocks per chunk per stage (launcher; partial rows at slot + q * slots)
    int ksplit_b;
    int dbg;                    // diagnostics (EXL3_MOE_COOP_DBG): 1 skip A epilogue, 2 skip B reduction
    int* runs;                  // run table (bsz > 1): [n_runs, -, run_start[slots_max + 1], order[slots_max]]
    int slots_max;
    int rows_max;               // token rows the output scratch holds
    int n_local;                // entries in the pointer tables
    int sh_gate_n;              // shared gate weight length (checked against H per call)

    // Output
    float* out;                 // (bsz, H_out), row stride out_stride
    int out_stride;

    // Shared expert (nullable): out += gate * sh_out, gate = sigmoid(x . sh_gate_w) or 1
    const float* sh_out;        // (bsz, H), row stride H
    const half* sh_gate_w;      // (H,) or null
};

// Launch with plain device pointers (called from BC_BlockSparseMLP). K: gate/up and down bit
// widths (gate and up always share one, the converter allocates them as one group); cb: 0 default,
// 1 mcg, 2 mul1 codebook, uniform across the three projections
void exl3_moe_coop_launch(const MoeCoopParams& p, float K_gu, float K_d, int cb, int device, cudaStream_t stream);

// Length of the int32 scratch for a module: completion counters + run table (slots_max =
// MAX_BSZN * top_k)
inline int64_t exl3_moe_coop_ctr_len(int64_t slots_max, int64_t bsz_max, int64_t I, int64_t Ho)
{
    return slots_max * (I / 128) + bsz_max * (Ho / 128) + 2 + (slots_max + 1) + slots_max;
}
