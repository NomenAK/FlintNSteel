// tests/swap_parity.cu - byte-exact VRAM / mapped pinned host slot exchange.
// fns-swap-parity --model F.gguf
#include "fns/model.h"
#include "../src/kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { std::fprintf(stderr, "%s : %s\n", #x, cudaGetErrorString(err_)); std::exit(1); } } while (0)

int main(int argc, char ** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--model") != 0) {
        std::fprintf(stderr, "usage: fns-swap-parity --model F.gguf\n");
        return 2;
    }
    fns::Model model(argv[2]);
    std::vector<size_t> sizes = {0, 1, 15, 16, 17, 31, 32, 33, 4095, 4096, 4097};
    for (const auto & layer : model.layers) {
        if (layer.dense) continue;
        const size_t ne = (size_t) model.hp.n_expert;
        const size_t bytes = layer.gate_exps.nbytes / ne + layer.up_exps.nbytes / ne + layer.down_exps.nbytes / ne;
        sizes.push_back((bytes + 255) / 256 * 256);
    }
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    cudaStream_t st = nullptr;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    uint32_t state = 12345;
    for (size_t n : sizes) {
        // Misalign either or both addresses as well as the aligned slot path; guard both ends.
        for (size_t aoff : {size_t(16), size_t(17)}) for (size_t boff : {size_t(16), size_t(17)}) {
            const size_t total = n + 34;
            uint8_t * dev = nullptr, * host = nullptr, * alias = nullptr;
            CK(cudaMalloc(&dev, total));
            CK(cudaHostAlloc(&host, total, cudaHostAllocMapped | cudaHostAllocWriteCombined));
            CK(cudaHostGetDevicePointer((void **) &alias, host, 0));
            std::vector<uint8_t> original_dev(total), original_host(total), actual_dev(total);
            for (size_t i = 0; i < total; ++i) {
                state = state * 1664525u + 1013904223u;
                original_dev[i] = (uint8_t) (state >> 24);
                state = state * 1664525u + 1013904223u;
                original_host[i] = (uint8_t) (state >> 24);
            }
            std::memcpy(host, original_host.data(), total);
            CK(cudaMemcpyAsync(dev, original_dev.data(), total, cudaMemcpyHostToDevice, st));
            for (int pass = 0; pass < 2; ++pass) {
                fns::k::swap_bytes(dev + aoff, alias + boff, n, st);
                CK(cudaStreamSynchronize(st));
                CK(cudaMemcpy(actual_dev.data(), dev, total, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < total; ++i) {
                    const bool in_dev = i >= aoff && i - aoff < n;
                    const bool in_host = i >= boff && i - boff < n;
                    const uint8_t expected_dev = pass == 0 && in_dev ? original_host[boff + i - aoff] : original_dev[i];
                    const uint8_t expected_host = pass == 0 && in_host ? original_dev[aoff + i - boff] : original_host[i];
                    if (actual_dev[i] != expected_dev || host[i] != expected_host) {
                        std::fprintf(stderr, "swap mismatch size=%zu dev_offset=%zu host_offset=%zu pass=%d byte=%zu\n",
                                     n, aoff, boff, pass, i);
                        return 1;
                    }
                }
            }
            CK(cudaFree(dev));
            CK(cudaFreeHost(host));
        }
    }
    CK(cudaStreamDestroy(st));
    std::printf("swap parity: %zu sizes passed\n", sizes.size());
    return 0;
}
