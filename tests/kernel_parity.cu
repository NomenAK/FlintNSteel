// tests/kernel_parity.cu - quantized GEMV parity on real GGUF blocks.
//
//   fns-kernel-parity --model F.gguf --x x.f32 --out dir
//
// For one tensor of each present format, applies fns::mmvq (and fns::mmvq_ids for experts, with 6 input rows
// like `down`) to a fixed vector x and writes the raw result to dir/<tensor>.f32. The reference is computed by
// tests/kernel_parity.py: dequantization with gguf-py (same llama.cpp repository), then a float64 product.
// What the test checks: our arguments (strides, channels, ids) match ggml's; the remaining error is that of the
// Q8_1 quantization of x.
#include "fns/ggml_kernels.h"
#include "fns/model.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); return 1; } } while (0)

int main(int argc, char ** argv) {
    const char * model_path = nullptr, * x_path = nullptr, * out_dir = nullptr;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--model")) model_path = argv[i + 1];
        else if (!std::strcmp(argv[i], "--x")) x_path = argv[i + 1];
        else if (!std::strcmp(argv[i], "--out")) out_dir = argv[i + 1];
    }
    if (!model_path || !x_path || !out_dir) {
        std::fprintf(stderr, "usage: fns-kernel-parity --model F.gguf --x x.f32 --out dir\n");
        return 2;
    }
    std::ifstream xf(x_path, std::ios::binary);
    std::vector<float> x(6 * 6144);
    xf.read((char *) x.data(), (std::streamsize) (x.size() * sizeof(float)));
    if (!xf) {
        std::fprintf(stderr, "x.f32 must contain 6*6144 float32 values\n");
        return 2;
    }

    fns::Model m(model_path);
    // One dense and one expert tensor per format, in layer order.
    std::map<std::string, const fns::Tensor *> dense, experts;
    auto take = [](std::map<std::string, const fns::Tensor *> & by, const fns::Tensor & t) {
        if (t.data && !by.count(ggml_type_name(t.type))) by[ggml_type_name(t.type)] = &t;
    };
    take(dense, m.output);
    for (const fns::Layer & L : m.layers) {
        for (const fns::Tensor * t : {&L.wq, &L.wk, &L.wo, &L.ffn_down, &L.down_shexp, &L.gate_shexp}) take(dense, *t);
        for (const fns::Tensor * t : {&L.gate_exps, &L.down_exps}) take(experts, *t);
    }

    float * dx = nullptr, * dy = nullptr;
    void * dxq = nullptr, * dw = nullptr;
    int32_t * dids = nullptr;
    CK(cudaMalloc(&dx, x.size() * sizeof(float)));
    CK(cudaMalloc(&dy, 6 * 150144 * sizeof(float)));
    CK(cudaMalloc(&dxq, 1 << 20));
    CK(cudaMalloc(&dids, 6 * sizeof(int32_t)));
    CK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    const int32_t ids[6] = {0, 255, 17, 128, 3, 200};
    CK(cudaMemcpy(dids, ids, sizeof(ids), cudaMemcpyHostToDevice));

    auto write = [&](const std::string & name, size_t n) -> bool {
        std::vector<float> y(n);
        if (cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        FILE * f = std::fopen((std::string(out_dir) + "/" + name + ".f32").c_str(), "wb");
        if (!f) return false;
        std::fwrite(y.data(), sizeof(float), n, f);
        std::fclose(f);
        std::printf("%s\n", name.c_str());
        return true;
    };

    for (const auto & [tname, t] : dense) {
        const int64_t ncols = t->ne[0], nrows = t->ne[1];
        CK(cudaMalloc(&dw, t->nbytes + 4096));
        CK(cudaMemset(dw, 0, t->nbytes + 4096));
        CK(cudaMemcpy(dw, t->data, t->nbytes, cudaMemcpyHostToDevice));
        fns::quantize_q8_1(dx, dxq, ncols, 1, nullptr);
        fns::mmvq(t->type, dw, ncols, nrows, dxq, dy, 1, nullptr);
        CK(cudaDeviceSynchronize());
        if (!write(t->name, (size_t) nrows)) return 1;
        CK(cudaFree(dw));
    }
    for (const auto & [tname, t] : experts) {
        const int64_t ncols = t->ne[0], nrows = t->ne[1], n_mats = t->ne[2];
        CK(cudaMalloc(&dw, t->nbytes + 4096));
        CK(cudaMemset(dw, 0, t->nbytes + 4096));
        CK(cudaMemcpy(dw, t->data, t->nbytes, cudaMemcpyHostToDevice));
        // One shared input row (gate/up), then six rows (down): the two engine call shapes.
        fns::quantize_q8_1(dx, dxq, ncols, 1, nullptr);
        fns::mmvq_ids(t->type, dw, ncols, nrows, n_mats, dxq, 1, dids, 6, dy, nullptr);
        CK(cudaDeviceSynchronize());
        if (!write(t->name + ".x1", (size_t) 6 * nrows)) return 1;
        fns::quantize_q8_1(dx, dxq, ncols, 6, nullptr);
        fns::mmvq_ids(t->type, dw, ncols, nrows, n_mats, dxq, 6, dids, 6, dy, nullptr);
        CK(cudaDeviceSynchronize());
        if (!write(t->name + ".x6", (size_t) 6 * nrows)) return 1;
        CK(cudaFree(dw));
    }
    return 0;
}
