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
Path: exllamav3/exllamav3_ext/quant/exl3_devctx.cuh
Modifications: no arithmetic changes.
*/
#pragma once

#include <tuple>
#include <mutex>
#include "../arch.cuh"

// Max allowable output size, in tiles. Used to allocate global lock buffer per device for sync across threadblocks
#define MAX_TILES_C (1024 * 1024)
#define MAX_BARRIERS 1024
#define BARRIER_LOCKS_OFFSET MAX_TILES_C

// MoE expert scheduler state, after the barrier counters: [0] next ticket, [1] retired groups,
// [2 + group] ticket published to group. Self-resetting, zero-initialized with the rest of the buffer
#define MOE_MAX_GROUPS 64
#define MOE_SCHED_OFFSET (MAX_TILES_C + 2 * MAX_BARRIERS)
#define MOE_SCHED_INTS (2 + MOE_MAX_GROUPS)
#if defined(USE_ROCM)
    // Whole-grid barrier of the EXL3 GEMM kernels (counter + sense), which ROCm launches without the
    // cooperative launch (see EXL3_COOP_LAUNCH in coop_autotune.cuh)
    #define EXL3_GRID_BARRIER_OFFSET (MOE_SCHED_OFFSET + MOE_SCHED_INTS)
    #define EXL3_GRID_BARRIER_INTS 2
#endif

// Workspace size
#define WORKSPACE_SIZE (16*1024*1024)

#define MAX_DEVICES 16
#define CC_OLD        1
#define CC_AMPERE     2
#define CC_ADA        3
#define CC_HOPPER     4
#define CC_BLACKWELL  5

// Singleton to manage context for each device. Stores device attributes and a large-enough lock buffer per device
class DevCtx
{
private:
    int num_sms[MAX_DEVICES] = {};
    int cc[MAX_DEVICES] = {};
    int smem_max[MAX_DEVICES] = {};
    void* locks[MAX_DEVICES] = {};
    void* ws[MAX_DEVICES] = {};
    std::mutex mtx;

public:
    static DevCtx& instance();
    int get_num_sms(int device);
    int get_cc(int device);
    // Device capability: dynamic shared memory per block the driver will grant, unclamped
    // (sm_86 reports 99 KB, Turing 64 KB). Use this to decide what a device can do.
    int get_smem_max(int device);
    // What an EXL3 kernel may actually request: the above, capped at the SMEM_MAX the kernels
    // are written against. Use this for cudaFuncSetAttribute and launch parameters.
    int get_smem_request(int device);
    void* get_ws(int device);
    int* get_locks(int device);

private:
    DevCtx() = default;
    DevCtx(const DevCtx&) = delete;
    DevCtx& operator=(const DevCtx&) = delete;
};

int g_get_cc(int device);
int g_get_num_sms(int device);
int g_get_smem_max(int device);

void prepare_ctx(int device);