
// gemm_at_bt_neon.cpp
// C(M,N) = A(M,K) * B(N,K)  —— 等价于 A * B^T
// 数据均为 row-major，类型 float (float32)
// 编译：g++ -O3 -ffast-math -march=armv8-a+simd gemm_at_bt_neon.cpp -o gemm

#include <arm_neon.h>
#include <cstddef>
#include <cstring>

// 为了可读性：索引宏（row-major）
#define A_AT(A, M, K, i, k) (A[(size_t)(i)*(K) + (k)])
#define B_AT(B, N, K, j, k) (B[(size_t)(j)*(K) + (k)])
#define C_AT(C, M, N, i, j) (C[(size_t)(i)*(N) + (j)])

// 8x4 微内核：计算 C[i..i+7, j..j+3] += A[i..i+7, :] * B[j..j+3, :]^T
static inline void micro_kernel_8x4_f32(
    const float* __restrict A, const float* __restrict B, float* __restrict C,
    int M, int N, int K, int i, int j)
{
    // 32 个向量累加器（每个是“4 段部分和”）
    float32x4_t acc[8][4];
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 4; ++c)
            acc[r][c] = vdupq_n_f32(0.0f);

    // 对于 B 的 4 行指针（行主序）
    const float* bptr[4] = {
        &B_AT(B, N, K, j + 0, 0),
        &B_AT(B, N, K, j + 1, 0),
        &B_AT(B, N, K, j + 2, 0),
        &B_AT(B, N, K, j + 3, 0),
    };

    // 对于 A 的 8 行指针
    const float* aptr[8];
    for (int r = 0; r < 8; ++r) {
        aptr[r] = &A_AT(A, M, K, i + r, 0);
    }

    // k 维度按 4 展开
    for (int k = 0; k < K; k += 4) {
        // 载入 B 的 4 行（各自 k..k+3）—— 连续内存
        float32x4_t b0 = vld1q_f32(bptr[0] + k);
        float32x4_t b1 = vld1q_f32(bptr[1] + k);
        float32x4_t b2 = vld1q_f32(bptr[2] + k);
        float32x4_t b3 = vld1q_f32(bptr[3] + k);

        // 对 A 的 8 行分别做向量乘加（与 B 的每一行做点积的“分段和”）
        for (int r = 0; r < 8; ++r) {
            float32x4_t a = vld1q_f32(aptr[r] + k);
            acc[r][0] = vmlaq_f32(acc[r][0], a, b0);
            acc[r][1] = vmlaq_f32(acc[r][1], a, b1);
            acc[r][2] = vmlaq_f32(acc[r][2], a, b2);
            acc[r][3] = vmlaq_f32(acc[r][3], a, b3);
        }
    }

    // 水平求和并写回 C
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 4; ++c) {
            float sum = vaddvq_f32(acc[r][c]); // 水平加总 4 段部分和
            C_AT(C, M, N, i + r, j + c) = sum;
        }
    }
}

// 4x4 微内核（用于 M 尾部不足 8 行但 >=4 行）
static inline void micro_kernel_4x4_f32(
    const float* __restrict A, const float* __restrict B, float* __restrict C,
    int M, int N, int K, int i, int j)
{
    float32x4_t acc[4][4];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            acc[r][c] = vdupq_n_f32(0.0f);

    const float* bptr[4] = {
        &B_AT(B, N, K, j + 0, 0),
        &B_AT(B, N, K, j + 1, 0),
        &B_AT(B, N, K, j + 2, 0),
        &B_AT(B, N, K, j + 3, 0),
    };
    const float* aptr[4] = {
        &A_AT(A, M, K, i + 0, 0),
        &A_AT(A, M, K, i + 1, 0),
        &A_AT(A, M, K, i + 2, 0),
        &A_AT(A, M, K, i + 3, 0),
    };

    for (int k = 0; k < K; k += 4) {
        float32x4_t b0 = vld1q_f32(bptr[0] + k);
        float32x4_t b1 = vld1q_f32(bptr[1] + k);
        float32x4_t b2 = vld1q_f32(bptr[2] + k);
        float32x4_t b3 = vld1q_f32(bptr[3] + k);

        for (int r = 0; r < 4; ++r) {
            float32x4_t a = vld1q_f32(aptr[r] + k);
            acc[r][0] = vmlaq_f32(acc[r][0], a, b0);
            acc[r][1] = vmlaq_f32(acc[r][1], a, b1);
            acc[r][2] = vmlaq_f32(acc[r][2], a, b2);
            acc[r][3] = vmlaq_f32(acc[r][3], a, b3);
        }
    }

    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            C_AT(C, M, N, i + r, j + c) = vaddvq_f32(acc[r][c]);
}

// scalar 收尾（行数 <4 或其他小尾巴）
static inline void kernel_scalar_f32(
    const float* __restrict A, const float* __restrict B, float* __restrict C,
    int M, int N, int K, int i0, int i1, int j0, int j1)
{
    for (int i = i0; i < i1; ++i) {
        for (int j = j0; j < j1; ++j) {
            const float* a = &A_AT(A, M, K, i, 0);
            const float* b = &B_AT(B, N, K, j, 0);
            float sum = 0.f;
            // K 是 512，可用简单的 4 次展开
            int k = 0;
            for (; k + 3 < K; k += 4) {
                sum += a[k+0] * b[k+0]
                     + a[k+1] * b[k+1]
                     + a[k+2] * b[k+2]
                     + a[k+3] * b[k+3];
            }
            for (; k < K; ++k) sum += a[k] * b[k];
            C_AT(C, M, N, i, j) = sum;
        }
    }
}

// 主函数：单线程 GEMM（A: MxK, B: NxK, C: MxN，全 row-major）
void gemm_at_bt_f32_neon(
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int M, int N, int K)
{
    // 可选：如果希望明确初始化 C（这里直接写满所有元素，不必先清零）

    const int mr = 8;
    const int nr = 4; // N=80 正好整除

    int j = 0;
    for (; j + nr - 1 < N; j += nr) {
        int i = 0;
        for (; i + mr - 1 < M; i += mr) {
            micro_kernel_8x4_f32(A, B, C, M, N, K, i, j);
        }
        // M 尾部：先尝试 4 行
        for (; i + 4 - 1 < M; i += 4) {
            micro_kernel_4x4_f32(A, B, C, M, N, K, i, j);
        }
        // 余下的少量行用标量
        if (i < M) {
            kernel_scalar_f32(A, B, C, M, N, K, i, M, j, j + nr);
        }
    }

    // 若 N 不是 4 的倍数，可在此用 2、1 列微核/标量兜底（本题 N=80 不需要）
}

#undef A_AT
#undef B_AT
#undef C_AT

// ====== 示例用法（可自行删除）======
#include <vector>
#include <iostream>
int main() {
    int M=900,N=80,K=512;
    std::vector<float> A(M*K), B(N*K), C(M*N);
    // 初始化 A、B...
    for (int i=0;i<M*K;i++) A[i] = (i%7)*0.1f;
    for (int i=0;i<N*K;i++) B[i] = (i%5)*0.2f;
    gemm_at_bt_f32_neon(A.data(), B.data(), C.data(), M, N, K);
    std::cout << C[0] << "\n";
    return 0;
}
