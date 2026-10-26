// src/ggml/ggml_cuda_host.cu - host-side ggml-cuda symbols referenced by mmvq.cu and quantize.cu, so they
// compile without the rest of ggml-cuda.cu.
//
// Derived from Strata src/prefill/ggml_cuda_host.cu (011c9e600a93bbc9fd451c563501b5831c32bbca, MIT, Copyright (c)
// 2026 Niko1221 and the Strata contributors), reduced to CUDA. The ggml_cuda_device_info fields follow
// ggml/src/ggml-cuda/ggml-cuda.cu:301-392 of llama.cpp 19ab5a5e (MIT, The ggml authors).
#include "common.cuh"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    std::fprintf(stderr, "ggml-cuda (fns): %s: %s\n  in %s at %s:%d\n", msg, stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

void ggml_cuda_set_device(int device) {
    int current = 0;
    CUDA_CHECK(cudaGetDevice(&current));
    if (current != device) {
        CUDA_CHECK(cudaSetDevice(device));
    }
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        in.physical_device_count = n;
        for (int id = 0; id < n; ++id) {
            auto attr = [id](cudaDeviceAttr a) {
                int v = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&v, a, id));
                return v;
            };
            auto & d = in.devices[id];
            d.cc = 100 * attr(cudaDevAttrComputeCapabilityMajor) + 10 * attr(cudaDevAttrComputeCapabilityMinor);
            d.nsm = attr(cudaDevAttrMultiProcessorCount);
            d.smpb = (size_t) attr(cudaDevAttrMaxSharedMemoryPerBlock);
            d.smpbo = (size_t) attr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
            d.warp_size = attr(cudaDevAttrWarpSize);
            d.integrated = false;
            d.supports_cooperative_launch = attr(cudaDevAttrCooperativeLaunch) != 0;
            d.vmm = false;
            size_t free_b = 0, total_b = 0;
            cudaSetDevice(id);
            cudaMemGetInfo(&free_b, &total_b);
            d.total_vram = total_b;
            d.physical_device = id;
            d.physical_share_count = 1;
            d.virtual_index = 0;
        }
        return in;
    }();
    return info;
}

namespace {
// Buffer pool for MMQ and FlashAttention (stream-k fixup tiles, temporary buffers). Buffers are kept and reused,
// since ggml requests the same few sizes on every launch. Derived from Strata (same file, lines 116-145).
struct CachingPool : ggml_cuda_pool {
    struct Buf { void * p; size_t size; bool used; };
    std::vector<Buf> bufs;
    std::mutex mu;
    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (!b.used && b.size >= size) { b.used = true; *actual_size = b.size; return b.p; }
        void * p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, size));
        bufs.push_back({p, size, true});
        *actual_size = size;
        return p;
    }
    void free(void * ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (b.p == ptr) { b.used = false; return; }
    }
    ~CachingPool() override {
        for (auto & b : bufs) cudaFree(b.p);
    }
};
}  // namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(ggml_backend_cuda_context &, int, int) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool());
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
