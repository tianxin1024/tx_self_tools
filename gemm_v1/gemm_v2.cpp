#include <arm_neon.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

// ===================== Naive Baseline =====================
void gemm_baseline(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[j*K + k]; // B 是 N×K row-major
            }
            C[i*N + j] = sum;
        }
    }
}

// ===================== Optimized NEON (之前的核心) =====================
#define A_AT(A, M, K, i, k) (A[(size_t)(i)*(K) + (k)])
#define B_AT(B, N, K, j, k) (B[(size_t)(j)*(K) + (k)])
#define C_AT(C, M, N, i, j) (C[(size_t)(i)*(N) + (j)])

static inline void micro_kernel_8x4_f32(
    const float* __restrict A, const float* __restrict B, float* __restrict C,
    int M, int N, int K, int i, int j)
{
    float32x4_t acc[8][4];
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 4; ++c)
            acc[r][c] = vdupq_n_f32(0.0f);

    const float* bptr[4] = {
        &B_AT(B, N, K, j + 0, 0),
        &B_AT(B, N, K, j + 1, 0),
        &B_AT(B, N, K, j + 2, 0),
        &B_AT(B, N, K, j + 3, 0),
    };

    const float* aptr[8];
    for (int r = 0; r < 8; ++r) {
        aptr[r] = &A_AT(A, M, K, i + r, 0);
    }

    for (int k = 0; k < K; k += 4) {
        float32x4_t b0 = vld1q_f32(bptr[0] + k);
        float32x4_t b1 = vld1q_f32(bptr[1] + k);
        float32x4_t b2 = vld1q_f32(bptr[2] + k);
        float32x4_t b3 = vld1q_f32(bptr[3] + k);

        for (int r = 0; r < 8; ++r) {
            float32x4_t a = vld1q_f32(aptr[r] + k);
            acc[r][0] = vmlaq_f32(acc[r][0], a, b0);
            acc[r][1] = vmlaq_f32(acc[r][1], a, b1);
            acc[r][2] = vmlaq_f32(acc[r][2], a, b2);
            acc[r][3] = vmlaq_f32(acc[r][3], a, b3);
        }
    }

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 4; ++c) {
            float sum = vaddvq_f32(acc[r][c]);
            C_AT(C, M, N, i + r, j + c) = sum;
        }
    }
}

static inline void kernel_scalar_f32(
    const float* __restrict A, const float* __restrict B, float* __restrict C,
    int M, int N, int K, int i0, int i1, int j0, int j1)
{
    for (int i = i0; i < i1; ++i) {
        for (int j = j0; j < j1; ++j) {
            const float* a = &A_AT(A, M, K, i, 0);
            const float* b = &B_AT(B, N, K, j, 0);
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += a[k] * b[k];
            }
            C_AT(C, M, N, i, j) = sum;
        }
    }
}

void gemm_at_bt_f32_neon(
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int M, int N, int K)
{
    const int mr = 8;
    const int nr = 4;

    int j = 0;
    for (; j + nr - 1 < N; j += nr) {
        int i = 0;
        for (; i + mr - 1 < M; i += mr) {
            micro_kernel_8x4_f32(A, B, C, M, N, K, i, j);
        }
        if (i < M) {
            kernel_scalar_f32(A, B, C, M, N, K, i, M, j, j + nr);
        }
    }
}

#undef A_AT
#undef B_AT
#undef C_AT

// ===================== 验证函数 =====================
bool check_result(const std::vector<float>& C1,
                  const std::vector<float>& C2,
                  int M, int N, float tol = 1e-4f)
{
    for (int i = 0; i < M*N; i++) {
        if (std::fabs(C1[i] - C2[i]) > tol) {
            std::cerr << "Mismatch at " << i
                      << " baseline=" << C1[i]
                      << " neon=" << C2[i]
                      << " diff=" << std::fabs(C1[i]-C2[i]) << "\n";
            return false;
        }
    }
    return true;
}

// ===================== 主函数 =====================
int main() {
    int M = 900, N = 80, K = 512;
    std::vector<float> A(M*K), B(N*K), C1(M*N), C2(M*N);

    // 随机初始化 A, B
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // baseline
    auto t0 = std::chrono::high_resolution_clock::now();
    gemm_baseline(A.data(), B.data(), C1.data(), M, N, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double baseline_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // neon
    auto t2 = std::chrono::high_resolution_clock::now();
    gemm_at_bt_f32_neon(A.data(), B.data(), C2.data(), M, N, K);
    auto t3 = std::chrono::high_resolution_clock::now();
    double neon_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // check
    bool ok = check_result(C1, C2, M, N, 1e-4f);

    std::cout << "Baseline time: " << baseline_ms << " ms\n";
    std::cout << "NEON opt time: " << neon_ms << " ms\n";
    std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << "\n";

    return 0;
}
