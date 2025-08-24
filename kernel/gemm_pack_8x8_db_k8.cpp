
#include <arm_neon.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

// =============== Baseline（用于校验） ===============
void gemm_baseline(const float* A, const float* B, float* C,
                   int M, int N, int K, float alpha)
{
    for (int i = 0; i < M; ++i) {
        const float* ai = A + (size_t)i*K;
        for (int j = 0; j < N; ++j) {
            const float* bj = B + (size_t)j*K; // B 是 NxK row-major
            float sum = 0.f;
            for (int k = 0; k < K; ++k) sum += ai[k] * bj[k];
            C[(size_t)i*N + j] = alpha * sum;
        }
    }
}

// =============== 工具：B 的 Kx8 面板打包 ===============
// 把 B 的 j..j+7 列打成 Kx8 连续面板：Bp[k*8 + c] = B[(j+c)*K + k]
static inline void pack_B_panel_8(const float* __restrict B, int N, int K,
                                  int j, float* __restrict Bp)
{
    const float* Bj[8];
    for (int c = 0; c < 8; ++c) {
        Bj[c] = B + (size_t)(j + c)*K;
    }
    for (int k = 0; k < K; ++k) {
        float* dst = Bp + (size_t)k*8;
        // 写成顺序有利于编译器向量化 store
        dst[0] = Bj[0][k];
        dst[1] = Bj[1][k];
        dst[2] = Bj[2][k];
        dst[3] = Bj[3][k];
        dst[4] = Bj[4][k];
        dst[5] = Bj[5][k];
        dst[6] = Bj[6][k];
        dst[7] = Bj[7][k];
    }
}

// =============== 8x8 微内核（配合 pack 后的 B 面板） ===============
// 计算：C[i..i+7, j..j+7] = alpha * A[i..i+7, :K] * B[j..j+7, :K]^T
// A：MxK row-major（传入的是 i 行起始处指针）
// Bp：Kx8 面板（连续），由 pack_B_panel_8 生成
// 一次处理 8 个 k 切片（k..k+7），内部做了“先载入 next，
// 再用 curr 计算”的双缓冲，并且用 vdupq_laneq_f32 来广播 4-lane，
// 减少标量抽取指令的依赖链。

static inline void micro_kernel_8x8_packB_db_k8(
    const float* __restrict A, int K,
    const float* __restrict Bp, // Kx8 面板
    float* __restrict C, int N,
    float alpha)
{
    // 累加器：每行两段（前4列/后4列）
    float32x4_t accL[8], accR[8];
    for (int r = 0; r < 8; ++r) { accL[r] = vdupq_n_f32(0.f); accR[r] = vdupq_n_f32(0.f); }

    const float* arow[8];
    for (int r = 0; r < 8; ++r) arow[r] = A + (size_t)r*K;

    int k = 0;

    // ---------- 预载入 curr（k..k+3 & k+4..k+7） ----------
    auto load_b_4 = [&](int kk, float32x4_t& L0, float32x4_t& R0,
                                 float32x4_t& L1, float32x4_t& R1,
                                 float32x4_t& L2, float32x4_t& R2,
                                 float32x4_t& L3, float32x4_t& R3) {
        const float* bp0 = Bp + (size_t)(kk + 0)*8;
        const float* bp1 = Bp + (size_t)(kk + 1)*8;
        const float* bp2 = Bp + (size_t)(kk + 2)*8;
        const float* bp3 = Bp + (size_t)(kk + 3)*8;
        L0 = vld1q_f32(bp0 + 0); R0 = vld1q_f32(bp0 + 4);
        L1 = vld1q_f32(bp1 + 0); R1 = vld1q_f32(bp1 + 4);
        L2 = vld1q_f32(bp2 + 0); R2 = vld1q_f32(bp2 + 4);
        L3 = vld1q_f32(bp3 + 0); R3 = vld1q_f32(bp3 + 4);
    };

    float32x4_t b0L,b0R,b1L,b1R,b2L,b2R,b3L,b3R; // curr(0..3)
    float32x4_t b4L,b4R,b5L,b5R,b6L,b6R,b7L,b7R; // curr(4..7)

    if (k + 7 < K) {
        load_b_4(k+0, b0L,b0R,b1L,b1R,b2L,b2R,b3L,b3R);
        load_b_4(k+4, b4L,b4R,b5L,b5R,b6L,b6R,b7L,b7R);
    }

    // ---------- 主循环：每次处理 8 个 k 切片 ----------
    for (; k + 7 < K; k += 8) {
        // 预载入 next（k+8..k+15），用临时寄存器缓存，等会交换
        float32x4_t nb0L,nb0R,nb1L,nb1R,nb2L,nb2R,nb3L,nb3R;
        float32x4_t nb4L,nb4R,nb5L,nb5R,nb6L,nb6R,nb7L,nb7R;
        if (k + 15 < K) {
            load_b_4(k+8,  nb0L,nb0R,nb1L,nb1R,nb2L,nb2R,nb3L,nb3R);
            load_b_4(k+12, nb4L,nb4R,nb5L,nb5R,nb6L,nb6R,nb7L,nb7R);
        }

        // 用 curr(0..7) 做 FMA
        for (int r = 0; r < 8; ++r) {
            // A[r] 的 8 个值：分两次加载 4+4
            float32x4_t a0_3 = vld1q_f32(arow[r] + k + 0);
            float32x4_t a4_7 = vld1q_f32(arow[r] + k + 4);

            // a0..a3
            float32x4_t a0 = vdupq_laneq_f32(a0_3, 0);
            float32x4_t a1 = vdupq_laneq_f32(a0_3, 1);
            float32x4_t a2 = vdupq_laneq_f32(a0_3, 2);
            float32x4_t a3 = vdupq_laneq_f32(a0_3, 3);
            accL[r] = vmlaq_f32(accL[r], a0, b0L); accR[r] = vmlaq_f32(accR[r], a0, b0R);
            accL[r] = vmlaq_f32(accL[r], a1, b1L); accR[r] = vmlaq_f32(accR[r], a1, b1R);
            accL[r] = vmlaq_f32(accL[r], a2, b2L); accR[r] = vmlaq_f32(accR[r], a2, b2R);
            accL[r] = vmlaq_f32(accL[r], a3, b3L); accR[r] = vmlaq_f32(accR[r], a3, b3R);

            // a4..a7
            float32x4_t a4 = vdupq_laneq_f32(a4_7, 0);
            float32x4_t a5 = vdupq_laneq_f32(a4_7, 1);
            float32x4_t a6 = vdupq_laneq_f32(a4_7, 2);
            float32x4_t a7 = vdupq_laneq_f32(a4_7, 3);
            accL[r] = vmlaq_f32(accL[r], a4, b4L); accR[r] = vmlaq_f32(accR[r], a4, b4R);
            accL[r] = vmlaq_f32(accL[r], a5, b5L); accR[r] = vmlaq_f32(accR[r], a5, b5R);
            accL[r] = vmlaq_f32(accL[r], a6, b6L); accR[r] = vmlaq_f32(accR[r], a6, b6R);
            accL[r] = vmlaq_f32(accR[r], a7, b7L); accR[r] = vmlaq_f32(accR[r], a7, b7R);
        }

        // 交换：curr <- next
        if (k + 15 < K) {
            b0L=nb0L; b0R=nb0R; b1L=nb1L; b1R=nb1R; b2L=nb2L; b2R=nb2R; b3L=nb3L; b3R=nb3R;
            b4L=nb4L; b4R=nb4R; b5L=nb5L; b5R=nb5R; b6L=nb6L; b6R=nb6R; b7L=nb7L; b7R=nb7R;
        }
    }

    // 处理剩余（k 尾巴 <8，K=512 恰好整除时不会进）
    for (; k < K; ++k) {
        float32x4_t bL = vld1q_f32(Bp + (size_t)k*8 + 0);
        float32x4_t bR = vld1q_f32(Bp + (size_t)k*8 + 4);
        for (int r = 0; r < 8; ++r) {
            float32x4_t av = vdupq_n_f32(arow[r][k]);
            accL[r] = vmlaq_f32(accL[r], av, bL);
            accR[r] = vmlaq_f32(accR[r], av, bR);
        }
    }

    // 写回 * alpha
    for (int r = 0; r < 8; ++r) {
        float* crow = C + (size_t)r*N;
        vst1q_f32(crow + 0, vmulq_n_f32(accL[r], alpha));
        vst1q_f32(crow + 4, vmulq_n_f32(accR[r], alpha));
    }
}


// =============== 4x8 微内核（M 尾部不足 8 行时用） ===============
static inline void micro_kernel_4x8_packB(
    const float* __restrict A, int K,
    const float* __restrict Bp, // Kx8
    float* __restrict C, int N,
    float alpha)
{
    float32x4_t accL[4], accR[4];
    for (int r = 0; r < 4; ++r) { accL[r] = vdupq_n_f32(0.f); accR[r] = vdupq_n_f32(0.f); }

    const float* arow[4];
    for (int r = 0; r < 4; ++r) arow[r] = A + (size_t)r*K;

    int k = 0;
    for (; k + 3 < K; k += 4) {
        const float* bp0 = Bp + (size_t)(k + 0)*8;
        const float* bp1 = Bp + (size_t)(k + 1)*8;
        const float* bp2 = Bp + (size_t)(k + 2)*8;
        const float* bp3 = Bp + (size_t)(k + 3)*8;

        float32x4_t b0L = vld1q_f32(bp0 + 0), b0R = vld1q_f32(bp0 + 4);
        float32x4_t b1L = vld1q_f32(bp1 + 0), b1R = vld1q_f32(bp1 + 4);
        float32x4_t b2L = vld1q_f32(bp2 + 0), b2R = vld1q_f32(bp2 + 4);
        float32x4_t b3L = vld1q_f32(bp3 + 0), b3R = vld1q_f32(bp3 + 4);

        for (int r = 0; r < 4; ++r) {
            float32x4_t a4 = vld1q_f32(arow[r] + k);
            float32x4_t a0 = vdupq_n_f32(vgetq_lane_f32(a4, 0));
            float32x4_t a1 = vdupq_n_f32(vgetq_lane_f32(a4, 1));
            float32x4_t a2 = vdupq_n_f32(vgetq_lane_f32(a4, 2));
            float32x4_t a3 = vdupq_n_f32(vgetq_lane_f32(a4, 3));

            accL[r] = vmlaq_f32(accL[r], a0, b0L);
            accR[r] = vmlaq_f32(accR[r], a0, b0R);
            accL[r] = vmlaq_f32(accL[r], a1, b1L);
            accR[r] = vmlaq_f32(accR[r], a1, b1R);
            accL[r] = vmlaq_f32(accL[r], a2, b2L);
            accR[r] = vmlaq_f32(accR[r], a2, b2R);
            accL[r] = vmlaq_f32(accL[r], a3, b3L);
            accR[r] = vmlaq_f32(accR[r], a3, b3R);
        }
    }

    for (; k < K; ++k) {
        float32x4_t bL = vld1q_f32(Bp + (size_t)k*8 + 0);
        float32x4_t bR = vld1q_f32(Bp + (size_t)k*8 + 4);
        for (int r = 0; r < 4; ++r) {
            float32x4_t av = vdupq_n_f32(arow[r][k]);
            accL[r] = vmlaq_f32(accL[r], av, bL);
            accR[r] = vmlaq_f32(accR[r], av, bR);
        }
    }

    for (int r = 0; r < 4; ++r) {
        float* crow = C + (size_t)r*N;
        float32x4_t sL = vmulq_n_f32(accL[r], alpha);
        float32x4_t sR = vmulq_n_f32(accR[r], alpha);
        vst1q_f32(crow + 0, sL);
        vst1q_f32(crow + 4, sR);
    }
}

// =============== 顶层：GEMM with B-pack & 8×8 kernel ===============
void gemm_pack8x8(const float* __restrict A,
                  const float* __restrict B,
                  float* __restrict C,
                  int M, int N, int K,
                  float alpha)
{
    // 为 B 的一个 8 列面板分配临时 buffer（K×8）
    std::vector<float> Bpanel((size_t)K * 8);

    // N 方向按 8 列一组打包 + 计算
    for (int j = 0; j + 7 < N; j += 8) {
        pack_B_panel_8(B, N, K, j, Bpanel.data());

        // M 方向按 8 行块
        int i = 0;
        for (; i + 7 < M; i += 8) {
            const float* Abase = A + (size_t)i*K;
            float* Cbase = C + (size_t)i*N + j;
            micro_kernel_8x8_packB_db_k8(Abase, K, Bpanel.data(), Cbase, N, alpha);
        }
        // 处理 4 行尾部（本问题 M=900 → 最后会有 4 行）
        if (i + 3 < M) {
            const float* Abase = A + (size_t)i*K;
            float* Cbase = C + (size_t)i*N + j;
            micro_kernel_4x8_packB(Abase, K, Bpanel.data(), Cbase, N, alpha);
            i += 4;
        }
        // 余下不足 4 行：标量兜底
        for (; i < M; ++i) {
            float* crow = C + (size_t)i*N + j;
            const float* ai = A + (size_t)i*K;
            for (int c = 0; c < 8; ++c) {
                float sum = 0.f;
                const float* bp = Bpanel.data() + c; // 注意：每个 k 步长是 8
                for (int k = 0; k < K; ++k) sum += ai[k] * bp[(size_t)k*8];
                crow[c] = alpha * sum;
            }
        }
    }

    // 若 N 不是 8 的倍数，这里可以再写 8→4→2→1 的尾处理（本题 N=80，正好整除，不需要）
}

// =============== 验证 & 计时 ===============
bool check_result(const std::vector<float>& X,
                  const std::vector<float>& Y,
                  int M, int N, float tol = 1e-4f)
{
    for (int i = 0; i < M*N; ++i) {
        if (std::fabs(X[i] - Y[i]) > tol) {
            std::cerr << "Mismatch at " << i
                      << " base=" << X[i]
                      << " test=" << Y[i]
                      << " diff=" << std::fabs(X[i]-Y[i]) << "\n";
            return false;
        }
    }
    return true;
}

template<class F>
double bench_avg_ms(F&& fn, int warmup = 2, int iters = 30) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
    const int M = 900, N = 80, K = 512;
    const float alpha = 1.0f; // 可改

    std::vector<float> A((size_t)M*K), B((size_t)N*K), Cbase((size_t)M*N), Copt((size_t)M*N);

    // 随机初始化
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // Baseline（单次计时）
    auto t0 = std::chrono::high_resolution_clock::now();
    gemm_baseline(A.data(), B.data(), Cbase.data(), M, N, K, alpha);
    auto t1 = std::chrono::high_resolution_clock::now();
    double baseline_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Pack+8x8（多次平均）
    auto run_opt = [&](){
        gemm_pack8x8(A.data(), B.data(), Copt.data(), M, N, K, alpha);
    };
    double opt_ms = bench_avg_ms(run_opt, 2, 30);

    bool ok = check_result(Cbase, Copt, M, N, 1e-4f);

    std::cout << "Alpha = " << alpha << "\n";
    std::cout << "Baseline time (1 run): " << baseline_ms << " ms\n";
    std::cout << "Pack+8x8 NEON (avg)  : " << opt_ms      << " ms\n";
    std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << "\n";
    std::cout << "Target < 8ms? " << (opt_ms < 8.0 ? "YES" : "NO") << "\n";
    return 0;
}
