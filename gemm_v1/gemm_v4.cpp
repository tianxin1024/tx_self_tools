// gemm_alpha_db.cpp
#include <arm_neon.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

#define A_AT(A, M, K, i, k) (A[(size_t)(i)*(K) + (k)])
#define B_AT(B, N, K, j, k) (B[(size_t)(j)*(K) + (k)])
#define C_AT(C, M, N, i, j) (C[(size_t)(i)*(N) + (j)])

// ===================== Naive Baseline =====================
void gemm_baseline(const float* A, const float* B, float* C,
                   int M, int N, int K, float alpha) {
    for (int i = 0; i < M; i++) {
        const float* ai = A + (size_t)i*K;
        for (int j = 0; j < N; j++) {
            const float* bj = B + (size_t)j*K;
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += ai[k] * bj[k];
            C[(size_t)i*N + j] = alpha * sum;
        }
    }
}

// ===================== NEON 基础 8x4 内核 =====================
static inline void micro_kernel_8x4_f32_basic(
    const float* __restrict A, const float* __restrict B, float* __restrict C,
    int M, int N, int K, int i, int j, float alpha)
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
    for (int r = 0; r < 8; ++r) aptr[r] = &A_AT(A, M, K, i + r, 0);

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

    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 4; ++c)
            C_AT(C, M, N, i + r, j + c) = alpha * vaddvq_f32(acc[r][c]);
}

// ===================== NEON 双缓冲 8x4 内核 =====================
// 思路：先载入 k=0 的 A/B（curr），每一轮预取/载入下一批（next），用 curr 计算后再交换。
static inline void micro_kernel_8x4_f32_db(
    const float* __restrict A, const float* __restrict B, float* __restrict C,
    int M, int N, int K, int i, int j, float alpha)
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
    for (int r = 0; r < 8; ++r) aptr[r] = &A_AT(A, M, K, i + r, 0);

    // 载入首批 curr
    float32x4_t b0 = vld1q_f32(bptr[0] + 0);
    float32x4_t b1 = vld1q_f32(bptr[1] + 0);
    float32x4_t b2 = vld1q_f32(bptr[2] + 0);
    float32x4_t b3 = vld1q_f32(bptr[3] + 0);

    float32x4_t a_curr[8];
    for (int r = 0; r < 8; ++r) a_curr[r] = vld1q_f32(aptr[r] + 0);

    // 主循环：k 跳到下一组，并预先载入 next，然后用 curr 进行计算
    for (int k = 4; k < K; k += 4) {
        // 可选软件预取（对不同芯片收益不同，可留可去）
        // for (int r = 0; r < 8; ++r) __builtin_prefetch(aptr[r] + k + 16, 0, 1);
        // for (int c = 0; c < 4; ++c) __builtin_prefetch(bptr[c] + k + 16, 0, 1);

        // 预取下一批 next
        float32x4_t nb0 = vld1q_f32(bptr[0] + k);
        float32x4_t nb1 = vld1q_f32(bptr[1] + k);
        float32x4_t nb2 = vld1q_f32(bptr[2] + k);
        float32x4_t nb3 = vld1q_f32(bptr[3] + k);

        float32x4_t a_next[8];
        for (int r = 0; r < 8; ++r) a_next[r] = vld1q_f32(aptr[r] + k);

        // 用 curr 计算
        for (int r = 0; r < 8; ++r) {
            acc[r][0] = vmlaq_f32(acc[r][0], a_curr[r], b0);
            acc[r][1] = vmlaq_f32(acc[r][1], a_curr[r], b1);
            acc[r][2] = vmlaq_f32(acc[r][2], a_curr[r], b2);
            acc[r][3] = vmlaq_f32(acc[r][3], a_curr[r], b3);
        }

        // 交换 curr <- next
        b0 = nb0; b1 = nb1; b2 = nb2; b3 = nb3;
        for (int r = 0; r < 8; ++r) a_curr[r] = a_next[r];
    }

    // 处理最后一批 curr
    for (int r = 0; r < 8; ++r) {
        acc[r][0] = vmlaq_f32(acc[r][0], a_curr[r], b0);
        acc[r][1] = vmlaq_f32(acc[r][1], a_curr[r], b1);
        acc[r][2] = vmlaq_f32(acc[r][2], a_curr[r], b2);
        acc[r][3] = vmlaq_f32(acc[r][3], a_curr[r], b3);
    }

    // 水平加并写回
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 4; ++c)
            C_AT(C, M, N, i + r, j + c) = alpha * vaddvq_f32(acc[r][c]);
}

// ===================== 标量尾部（行尾/其他收尾） =====================
static inline void kernel_scalar_f32(
    const float* __restrict A, const float* __restrict B, float* __restrict C,
    int M, int N, int K, int i0, int i1, int j0, int j1, float alpha)
{
    for (int i = i0; i < i1; ++i) {
        for (int j = j0; j < j1; ++j) {
            const float* a = &A_AT(A, M, K, i, 0);
            const float* b = &B_AT(B, N, K, j, 0);
            float sum = 0.f;
            for (int k = 0; k < K; ++k) sum += a[k] * b[k];
            C_AT(C, M, N, i, j) = alpha * sum;
        }
    }
}

// ===================== 两个外壳：基础 NEON 与 双缓冲 NEON =====================
void gemm_neon_basic(const float* __restrict A, const float* __restrict B,
                     float* __restrict C, int M, int N, int K, float alpha)
{
    const int mr = 8, nr = 4;
    int j = 0;
    for (; j + nr - 1 < N; j += nr) {
        int i = 0;
        for (; i + mr - 1 < M; i += mr) {
            micro_kernel_8x4_f32_basic(A, B, C, M, N, K, i, j, alpha);
        }
        if (i < M) kernel_scalar_f32(A, B, C, M, N, K, i, M, j, j + nr, alpha);
    }
}

void gemm_neon_db(const float* __restrict A, const float* __restrict B,
                  float* __restrict C, int M, int N, int K, float alpha)
{
    const int mr = 8, nr = 4;
    int j = 0;
    for (; j + nr - 1 < N; j += nr) {
        int i = 0;
        for (; i + mr - 1 < M; i += mr) {
            micro_kernel_8x4_f32_db(A, B, C, M, N, K, i, j, alpha);
        }
        if (i < M) kernel_scalar_f32(A, B, C, M, N, K, i, M, j, j + nr, alpha);
    }
}

// ===================== 验证函数 =====================
bool check_result(const std::vector<float>& C1,
                  const std::vector<float>& C2,
                  int M, int N, float tol = 1e-4f)
{
    for (int i = 0; i < M*N; i++) {
        if (std::fabs(C1[i] - C2[i]) > tol) {
            std::cerr << "Mismatch at " << i
                      << " base=" << C1[i]
                      << " test=" << C2[i]
                      << " diff=" << std::fabs(C1[i]-C2[i]) << "\n";
            return false;
        }
    }
    return true;
}

// ===================== 计时小工具 =====================
template<class F>
double bench_ms(F&& fn, int warmup = 2, int iters = 30) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// ===================== 主函数：对比基础 NEON 与 双缓冲 NEON =====================
int main() {
    int M = 900, N = 80, K = 512;
    float alpha = 1.0f;  // 可改

    std::vector<float> A(M*K), B(N*K), C_baseline(M*N), C_basic(M*N), C_db(M*N);

    // 随机初始化
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // 正确性基准：baseline
    double t_base = bench_ms([&](){ gemm_baseline(A.data(), B.data(), C_baseline.data(), M, N, K, alpha); }, 1, 1); // 只跑一次计时，避免太慢

    // 基础 NEON：计时与校验
    double t_basic = bench_ms([&](){ gemm_neon_basic(A.data(), B.data(), C_basic.data(), M, N, K, alpha); });
    bool ok_basic = check_result(C_baseline, C_basic, M, N, 1e-4f);

    // 双缓冲 NEON：计时与校验
    double t_db = bench_ms([&](){ gemm_neon_db(A.data(), B.data(), C_db.data(), M, N, K, alpha); });
    bool ok_db = check_result(C_baseline, C_db, M, N, 1e-4f);

    std::cout << "Alpha = " << alpha << "\n";
    std::cout << "Baseline  (naive)     : " << t_base  << " ms (single run)\n";
    std::cout << "NEON basic (8x4)      : " << t_basic << " ms (avg)\n";
    std::cout << "NEON double-buffer(8x4): " << t_db    << " ms (avg)\n";
    std::cout << "Correctness basic: " << (ok_basic ? "PASS" : "FAIL") << "\n";
    std::cout << "Correctness  d-buf: " << (ok_db ? "PASS" : "FAIL") << "\n";
    std::cout << "Target < 8ms? basic=" << (t_basic < 8.0 ? "YES" : "NO")
              << ", d-buf=" << (t_db < 8.0 ? "YES" : "NO") << "\n";
    return 0;
}
