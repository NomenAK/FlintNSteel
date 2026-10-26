// src/attention_mma.cu - global split-K attention, tensor-core F16 -> F32 QK/PV.
// Each warp handles 64 KV positions and two columns x eight GQA heads.
// The same per-row computation serves W=1 (second column masked) and W>1. Rounding differs from the FP32 path:
// explicit opt-in, disabled by default.
#include "kernels.cuh"

namespace fns::k {
namespace {
constexpr int D = 128, NH = 32, NKV = 4, G = 8, CHUNK = 64;

__device__ __forceinline__ unsigned hpair(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned *>(&h);
}
__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(128)
k_attention_mma_part(const float * q, const __half * K, const __half * V, float * po, float2 * pm,
                     const int32_t * pos, float scale, int q_stride, int part_stride, int cols, int chunks) {
    const int lane = threadIdx.x & 31, chunk = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int c0 = blockIdx.z * 2, c1 = c0 + 1, hk = blockIdx.y;
    const int p0 = *pos + c0, p1 = *pos + c1;
    const int first = chunk * CHUNK;
    if (chunk >= chunks || first > *pos + cols - 1) return;
    const int g = lane >> 2, q4 = lane & 3, head = hk * G + g;
    const float * q0 = q + (size_t) c0 * q_stride + head * D;
    const float * q1 = q + (size_t) (c1 < cols ? c1 : c0) * q_stride + head * D;
    unsigned qa[8][4];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int d = k * 16 + q4 * 2;
        qa[k][0] = hpair(q0[d] * scale, q0[d + 1] * scale);
        qa[k][1] = hpair(q1[d] * scale, q1[d + 1] * scale);
        qa[k][2] = hpair(q0[d + 8] * scale, q0[d + 9] * scale);
        qa[k][3] = hpair(q1[d + 8] * scale, q1[d + 9] * scale);
    }
    float out[16][4] = {};
    float m0 = -INFINITY, m1 = -INFINITY, sum0 = 0, sum1 = 0;
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int start = first + tile * 16;
        if (start > *pos + cols - 1) break;
        float s[2][4] = {};
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            // B is K transposed: each lane owns two feature pairs of one KV position.
            const int j = min(start + nt * 8 + g, *pos + cols - 1);
            const __half * kr = K + ((size_t) j * NKV + hk) * D;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int d = k * 16 + q4 * 2;
                mma(s[nt], qa[k], *reinterpret_cast<const unsigned *>(kr + d),
                    *reinterpret_cast<const unsigned *>(kr + d + 8));
            }
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int j = start + nt * 8 + q4 * 2;
            if (j > p0) s[nt][0] = -INFINITY;
            if (j + 1 > p0) s[nt][1] = -INFINITY;
            if (j > p1 || c1 >= cols) s[nt][2] = -INFINITY;
            if (j + 1 > p1 || c1 >= cols) s[nt][3] = -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        const float a0 = mx0 == -INFINITY ? 1.0f : expf(m0 - mx0);
        const float a1 = mx1 == -INFINITY ? 1.0f : expf(m1 - mx1);
        const float b0 = mx0 == -INFINITY ? 0.0f : mx0;
        const float b1 = mx1 == -INFINITY ? 0.0f : mx1;
        m0 = mx0; m1 = mx1;
        sum0 *= a0; sum1 *= a1;
        unsigned pa[4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const float e0 = expf(s[nt][0] - b0), e1 = expf(s[nt][1] - b0);
            const float e2 = expf(s[nt][2] - b1), e3 = expf(s[nt][3] - b1);
            sum0 += e0 + e1; sum1 += e2 + e3;
            pa[nt * 2] = hpair(e0, e1);
            pa[nt * 2 + 1] = hpair(e2, e3);
        }
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
            out[dn][0] *= a0; out[dn][1] *= a0;
            out[dn][2] *= a1; out[dn][3] *= a1;
            // B is V: contiguous dimensions across lane groups, two pairs of KV positions.
            const int d = dn * 8 + g;
            const int j0 = min(start + q4 * 2, *pos + cols - 1);
            const int j1 = min(start + q4 * 2 + 1, *pos + cols - 1);
            const int j2 = min(start + q4 * 2 + 8, *pos + cols - 1);
            const int j3 = min(start + q4 * 2 + 9, *pos + cols - 1);
            const unsigned v0 = hpair(__half2float(V[((size_t) j0 * NKV + hk) * D + d]),
                                     __half2float(V[((size_t) j1 * NKV + hk) * D + d]));
            const unsigned v1 = hpair(__half2float(V[((size_t) j2 * NKV + hk) * D + d]),
                                     __half2float(V[((size_t) j3 * NKV + hk) * D + d]));
            mma(out[dn], pa, v0, v1);
        }
    }
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
    float * o0 = po + (size_t) c0 * part_stride + ((size_t) head * chunks + chunk) * D;
    float * o1 = po + (size_t) c1 * part_stride + ((size_t) head * chunks + chunk) * D;
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
        const int d = dn * 8 + q4 * 2;
        o0[d] = out[dn][0]; o0[d + 1] = out[dn][1];
        if (c1 < cols) { o1[d] = out[dn][2]; o1[d + 1] = out[dn][3]; }
    }
    if (q4 == 0) {
        auto * ms0 = reinterpret_cast<float2 *>(reinterpret_cast<float *>(pm) + (size_t) c0 * part_stride);
        ms0[(size_t) head * chunks + chunk] = make_float2(m0, sum0);
        if (c1 < cols) {
            auto * ms1 = reinterpret_cast<float2 *>(reinterpret_cast<float *>(pm) + (size_t) c1 * part_stride);
            ms1[(size_t) head * chunks + chunk] = make_float2(m1, sum1);
        }
    }
}
}  // namespace

void attention_mma_part(const float * q, const void * kc, const void * vc, float * part_o, float2 * part_ms,
                        const int32_t * pos, float scale, int q_stride, int part_stride, int cols, int chunks,
                        cudaStream_t st) {
    k_attention_mma_part<<<dim3((chunks + 3) / 4, NKV, (cols + 1) / 2), 128, 0, st>>>(
        q, static_cast<const __half *>(kc), static_cast<const __half *>(vc), part_o, part_ms, pos, scale,
        q_stride, part_stride, cols, chunks);
}
}  // namespace fns::k
