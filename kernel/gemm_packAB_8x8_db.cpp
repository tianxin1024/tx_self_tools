#include <arm_neon.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>

// =============== 小工具 ===============
static inline float32x4_t dup_laneq0(float32x4_t v) { return vdupq_laneq_f32(v, 0); }
static inline float32x4_t dup_laneq1(float32x4_t v) { return vdupq_laneq_f32(v, 1); }
static inline float32x4_t dup_laneq2(float32x4_t v) { return vdupq_laneq_f32(v, 2); }
static inline float32x4_t dup_laneq3(float32x4_t v) { return vdupq_laneq_f32(v, 3); }

// =============== Baseline（校验用） ===============
void gemm_baseline(const float* A, const float* B, float* C,
                   int M, int N, int K, float alpha)
{
    for (int i = 0; i < M; ++i) {
        const float* ai = A + (size_t)i*K;
        for (int j = 0; j < N; ++j) {
            const float* bj = B + (size_t)j*K; // B: N×K row-major
            float sum = 0.f;
            for (int k = 0; k < K; ++k) sum += ai[k] * bj[k];
            C[(size_t)i*N + j] = alpha * sum;
        }
    }
}

// =============== B 的 K×8 面板打包 ===============
// Bp[k*8 + c] = B[(j+c)*K + k], 0<=c<cols
static inline void pack_B_panel_8(const float* __restrict B, int N, int K,
                                  int j, int cols, float* __restrict Bp)
{
    // cols: 本面板实际列数（<=8）
    const float* Bj[8];
    for (int c = 0; c < cols; ++c) Bj[c] = B + (size_t)(j + c)*K;

    for (int k = 0; k < K; ++k) {
        float* dst = Bp + (size_t)k*8;
        int c = 0;
        for (; c + 3 < cols; c += 4) {
            // 尽量让编译器向量化 store；但这里安全起见按标量写
            dst[c+0] = Bj[c+0][k];
            dst[c+1] = Bj[c+1][k];
            dst[c+2] = Bj[c+2][k];
            dst[c+3] = Bj[c+3][k];
        }
        for (; c < cols; ++c) dst[c] = Bj[c][k];
        // 对于 cols<8 的尾部，不必清零（微核不会读超出 cols 的列）
    }
}

// =============== A 的 K×8 / K×4 面板打包 ===============
// Ap[k*8 + r] = A[(i+r, k)]，r=0..rows-1
static inline void pack_A_panel_8(const float* __restrict A, int M, int K,
                                  int i, int rows, float* __restrict Ap)
{
    for (int k = 0; k < K; ++k) {
        for (int r = 0; r < rows; ++r) {
            Ap[(size_t)k*8 + r] = A[(size_t)(i + r)*K + k];
        }
    }
}
static inline void pack_A_panel_4(const float* __restrict A, int M, int K,
                                  int i, int rows, float* __restrict Ap)
{
    for (int k = 0; k < K; ++k) {
        for (int r = 0; r < rows; ++r) {
            Ap[(size_t)k*4 + r] = A[(size_t)(i + r)*K + k];
        }
    }
}

// =============== 8×8 微核（A、B 均已打包，k 展开 8 + 双缓冲） ===============
// Ap: K×8, Bp: K×8, 输出 C(8×8) 到 Cbase（行主序，leading dim = N）
static inline void micro_kernel_8x8_packAB_db_k8(
    const float* __restrict Ap, const float* __restrict Bp,
    float* __restrict Cbase, int N, int K, float alpha)
{
    float32x4_t accL[8], accR[8];
    for (int r = 0; r < 8; ++r) { accL[r] = vdupq_n_f32(0.f); accR[r] = vdupq_n_f32(0.f); }

    auto load_b4 = [&](int kk,
                       float32x4_t& L0, float32x4_t& R0,
                       float32x4_t& L1, float32x4_t& R1,
                       float32x4_t& L2, float32x4_t& R2,
                       float32x4_t& L3, float32x4_t& R3)
    {
        const float* bp0 = Bp + (size_t)(kk + 0)*8;
        const float* bp1 = Bp + (size_t)(kk + 1)*8;
        const float* bp2 = Bp + (size_t)(kk + 2)*8;
        const float* bp3 = Bp + (size_t)(kk + 3)*8;
        L0 = vld1q_f32(bp0 + 0); R0 = vld1q_f32(bp0 + 4);
        L1 = vld1q_f32(bp1 + 0); R1 = vld1q_f32(bp1 + 4);
        L2 = vld1q_f32(bp2 + 0); R2 = vld1q_f32(bp2 + 4);
        L3 = vld1q_f32(bp3 + 0); R3 = vld1q_f32(bp3 + 4);
    };

    // 预载 curr: k..k+7
    int k = 0;
    float32x4_t b0L,b0R,b1L,b1R,b2L,b2R,b3L,b3R;
    float32x4_t b4L,b4R,b5L,b5R,b6L,b6R,b7L,b7R;
    if (k + 7 < K) {
        load_b4(k+0, b0L,b0R,b1L,b1R,b2L,b2R,b3L,b3R);
        load_b4(k+4, b4L,b4R,b5L,b5R,b6L,b6R,b7L,b7R);
    }

    for (; k + 7 < K; k += 8) {
        // 预载 next（k+8..k+15）
        float32x4_t nb0L,nb0R,nb1L,nb1R,nb2L,nb2R,nb3L,nb3R;
        float32x4_t nb4L,nb4R,nb5L,nb5R,nb6L,nb6R,nb7L,nb7R;
        if (k + 15 < K) {
            load_b4(k+8,  nb0L,nb0R,nb1L,nb1R,nb2L,nb2R,nb3L,nb3R);
            load_b4(k+12, nb4L,nb4R,nb5L,nb5R,nb6L,nb6R,nb7L,nb7R);
        }

        // 用 curr(0..7) 计算：A 的每个 k 对应 8 个行值（连续）
        // 我们分两组处理：k..k+3 & k+4..k+7
        // ---- k..k+3
        for (int t = 0; t < 4; ++t) {
            const float* ap = Ap + (size_t)(k + t)*8; // 8 行的同一 k
            float32x4_t a01 = vld1q_f32(ap + 0);      // r0..r3
            float32x4_t a45 = vld1q_f32(ap + 4);      // r4..r7

            float32x4_t a0 = dup_laneq0(a01), a1 = dup_laneq1(a01);
            float32x4_t a2 = dup_laneq2(a01), a3 = dup_laneq3(a01);
            float32x4_t a4 = dup_laneq0(a45), a5 = dup_laneq1(a45);
            float32x4_t a6 = dup_laneq2(a45), a7 = dup_laneq3(a45);

            const float32x4_t BL = (t==0? b0L : t==1? b1L : t==2? b2L : b3L);
            const float32x4_t BR = (t==0? b0R : t==1? b1R : t==2? b2R : b3R);

            accL[0] = vmlaq_f32(accL[0], a0, BL);  accR[0] = vmlaq_f32(accR[0], a0, BR);
            accL[1] = vmlaq_f32(accL[1], a1, BL);  accR[1] = vmlaq_f32(accR[1], a1, BR);
            accL[2] = vmlaq_f32(accL[2], a2, BL);  accR[2] = vmlaq_f32(accR[2], a2, BR);
            accL[3] = vmlaq_f32(accL[3], a3, BL);  accR[3] = vmlaq_f32(accR[3], a3, BR);
            accL[4] = vmlaq_f32(accL[4], a4, BL);  accR[4] = vmlaq_f32(accR[4], a4, BR);
            accL[5] = vmlaq_f32(accL[5], a5, BL);  accR[5] = vmlaq_f32(accR[5], a5, BR);
            accL[6] = vmlaq_f32(accL[6], a6, BL);  accR[6] = vmlaq_f32(accR[6], a6, BR);
            accL[7] = vmlaq_f32(accL[7], a7, BL);  accR[7] = vmlaq_f32(accR[7], a7, BR);
        }

        // ---- k+4..k+7
        for (int t = 0; t < 4; ++t) {
            const float* ap = Ap + (size_t)(k + 4 + t)*8;
            float32x4_t a01 = vld1q_f32(ap + 0);
            float32x4_t a45 = vld1q_f32(ap + 4);

            float32x4_t a0 = dup_laneq0(a01), a1 = dup_laneq1(a01);
            float32x4_t a2 = dup_laneq2(a01), a3 = dup_laneq3(a01);
            float32x4_t a4 = dup_laneq0(a45), a5 = dup_laneq1(a45);
            float32x4_t a6 = dup_laneq2(a45), a7 = dup_laneq3(a45);

            const float32x4_t BL = (t==0? b4L : t==1? b5L : t==2? b6L : b7L);
            const float32x4_t BR = (t==0? b4R : t==1? b5R : t==2? b6R : b7R);

            accL[0] = vmlaq_f32(accL[0], a0, BL);  accR[0] = vmlaq_f32(accR[0], a0, BR);
            accL[1] = vmlaq_f32(accL[1], a1, BL);  accR[1] = vmlaq_f32(accR[1], a1, BR);
            accL[2] = vmlaq_f32(accL[2], a2, BL);  accR[2] = vmlaq_f32(accR[2], a2, BR);
            accL[3] = vmlaq_f32(accL[3], a3, BL);  accR[3] = vmlaq_f32(accR[3], a3, BR);
            accL[4] = vmlaq_f32(accL[4], a4, BL);  accR[4] = vmlaq_f32(accR[4], a4, BR);
            accL[5] = vmlaq_f32(accL[5], a5, BL);  accR[5] = vmlaq_f32(accR[5], a5, BR);
            accL[6] = vmlaq_f32(accL[6], a6, BL);  accR[6] = vmlaq_f32(accR[6], a6, BR);
            accL[7] = vmlaq_f32(accL[7], a7, BL);  accR[7] = vmlaq_f32(accR[7], a7, BR);
        }

        // 交换 curr <- next
        if (k + 15 < K) {
            b0L=nb0L; b0R=nb0R; b1L=nb1L; b1R=nb1R; b2L=nb2L; b2R=nb2R; b3L=nb3L; b3R=nb3R;
            b4L=nb4L; b4R=nb4R; b5L=nb5L; b5R=nb5R; b6L=nb6L; b6R=nb6R; b7L=nb7L; b7R=nb7R;
        }
    }

    // 尾部（K 非 8 倍数时；K=512 不会进）
    for (; k < K; ++k) {
        const float* bpk = Bp + (size_t)k*8;
        float32x4_t bL = vld1q_f32(bpk + 0), bR = vld1q_f32(bpk + 4);

        const float* apk = Ap + (size_t)k*8;
        float32x4_t a01 = vld1q_f32(apk + 0);
        float32x4_t a45 = vld1q_f32(apk + 4);

        float32x4_t a0 = dup_laneq0(a01), a1 = dup_laneq1(a01);
        float32x4_t a2 = dup_laneq2(a01), a3 = dup_laneq3(a01);
        float32x4_t a4 = dup_laneq0(a45), a5 = dup_laneq1(a45);
        float32x4_t a6 = dup_laneq2(a45), a7 = dup_laneq3(a45);

        accL[0] = vmlaq_f32(accL[0], a0, bL);  accR[0] = vmlaq_f32(accR[0], a0, bR);
        accL[1] = vmlaq_f32(accL[1], a1, bL);  accR[1] = vmlaq_f32(accR[1], a1, bR);
        accL[2] = vmlaq_f32(accL[2], a2, bL);  accR[2] = vmlaq_f32(accR[2], a2, bR);
        accL[3] = vmlaq_f32(accL[3], a3, bL);  accR[3] = vmlaq_f32(accR[3], a3, bR);
        accL[4] = vmlaq_f32(accL[4], a4, bL);  accR[4] = vmlaq_f32(accR[4], a4, bR);
        accL[5] = vmlaq_f32(accL[5], a5, bL);  accR[5] = vmlaq_f32(accR[5], a5, bR);
        accL[6] = vmlaq_f32(accL[6], a6, bL);  accR[6] = vmlaq_f32(accR[6], a6, bR);
        accL[7] = vmlaq_f32(accL[7], a7, bL);  accR[7] = vmlaq_f32(accR[7], a7, bR);
    }

    // 写回 * alpha
    for (int r = 0; r < 8; ++r) {
        float* crow = Cbase + (size_t)r*N;
        vst1q_f32(crow + 0, vmulq_n_f32(accL[r], alpha));
        vst1q_f32(crow + 4, vmulq_n_f32(accR[r], alpha));
    }
}

// =============== 4×8 尾核（A、B 均打包；K 展开 8 + 双缓冲） ===============
static inline void micro_kernel_4x8_packAB_db_k8(
    const float* __restrict Ap, const float* __restrict Bp,
    float* __restrict Cbase, int N, int K, float alpha)
{
    float32x4_t accL[4], accR[4];
    for (int r = 0; r < 4; ++r) { accL[r] = vdupq_n_f32(0.f); accR[r] = vdupq_n_f32(0.f); }

    auto load_b4 = [&](int kk,
                       float32x4_t& L0, float32x4_t& R0,
                       float32x4_t& L1, float32x4_t& R1,
                       float32x4_t& L2, float32x4_t& R2,
                       float32x4_t& L3, float32x4_t& R3)
    {
        const float* bp0 = Bp + (size_t)(kk + 0)*8;
        const float* bp1 = Bp + (size_t)(kk + 1)*8;
        const float* bp2 = Bp + (size_t)(kk + 2)*8;
        const float* bp3 = Bp + (size_t)(kk + 3)*8;
        L0 = vld1q_f32(bp0 + 0); R0 = vld1q_f32(bp0 + 4);
        L1 = vld1q_f32(bp1 + 0); R1 = vld1q_f32(bp1 + 4);
        L2 = vld1q_f32(bp2 + 0); R2 = vld1q_f32(bp2 + 4);
        L3 = vld1q_f32(bp3 + 0); R3 = vld1q_f32(bp3 + 4);
    };

    int k = 0;
    float32x4_t b0L,b0R,b1L,b1R,b2L,b2R,b3L,b3R;
    float32x4_t b4L,b4R,b5L,b5R,b6L,b6R,b7L,b7R;
    if (k + 7 < K) {
        load_b4(k+0, b0L,b0R,b1L,b1R,b2L,b2R,b3L,b3R);
        load_b4(k+4, b4L,b4R,b5L,b5R,b6L,b6R,b7L,b7R);
    }

    for (; k + 7 < K; k += 8) {
        float32x4_t nb0L,nb0R,nb1L,nb1R,nb2L,nb2R,nb3L,nb3R;
        float32x4_t nb4L,nb4R,nb5L,nb5R,nb6L,nb6R,nb7L,nb7R;
        if (k + 15 < K) {
            load_b4(k+8,  nb0L,nb0R,nb1L,nb1R,nb2L,nb2R,nb3L,nb3R);
            load_b4(k+12, nb4L,nb4R,nb5L,nb5R,nb6L,nb6R,nb7L,nb7R);
        }

        // k..k+3
        for (int t = 0; t < 4; ++t) {
            const float* ap = Ap + (size_t)(k + t)*4; // 4 行
            float32x4_t a = vld1q_f32(ap);           // r0..r3
            float32x4_t a0 = dup_laneq0(a);
            float32x4_t a1 = dup_laneq1(a);
            float32x4_t a2 = dup_laneq2(a);
            float32x4_t a3 = dup_laneq3(a);
            const float32x4_t BL = (t==0? b0L : t==1? b1L : t==2? b2L : b3L);
            const float32x4_t BR = (t==0? b0R : t==1? b1R : t==2? b2R : b3R);

            accL[0] = vmlaq_f32(accL[0], a0, BL);  accR[0] = vmlaq_f32(accR[0], a0, BR);
            accL[1] = vmlaq_f32(accL[1], a1, BL);  accR[1] = vmlaq_f32(accR[1], a1, BR);
            accL[2] = vmlaq_f32(accL[2], a2, BL);  accR[2] = vmlaq_f32(accR[2], a2, BR);
            accL[3] = vmlaq_f32(accL[3], a3, BL);  accR[3] = vmlaq_f32(accR[3], a3, BR);
        }

        // k+4..k+7
        for (int t = 0; t < 4; ++t) {
            const float* ap = Ap + (size_t)(k + 4 + t)*4;
            float32x4_t a = vld1q_f32(ap);
            float32x4_t a0 = dup_laneq0(a);
            float32x4_t a1 = dup_laneq1(a);
            float32x4_t a2 = dup_laneq2(a);
            float32x4_t a3 = dup_laneq3(a);
            const float32x4_t BL = (t==0? b4L : t==1? b5L : t==2? b6L : b7L);
            const float32x4_t BR = (t==0? b4R : t==1? b5R : t==2? b6R : b7R);

            accL[0] = vmlaq_f32(accL[0], a0, BL);  accR[0] = vmlaq_f32(accR[0], a0, BR);
            accL[1] = vmlaq_f32(accL[1], a1, BL);  accR[1] = vmlaq_f32(accR[1], a1, BR);
            accL[2] = vmlaq_f32(accL[2], a2, BL);  accR[2] = vmlaq_f32(accR[2], a2, BR);
            accL[3] = vmlaq_f32(accL[3], a3, BL);  accR[3] = vmlaq_f32(accR[3], a3, BR);
        }

        if (k + 15 < K) {
            b0L=nb0L; b0R=nb0R; b1L=nb1L; b1R=nb1R; b2L=nb2L; b2R=nb2R; b3L=nb3L; b3R=nb3R;
            b4L=nb4L; b4R=nb4R; b5L=nb5L; b5R=nb5R; b6L=nb6L; b6R=nb6R; b7L=nb7L; b7R=nb7R;
        }
    }

    for (; k < K; ++k) {
        const float* bpk = Bp + (size_t)k*8;
        float32x4_t bL = vld1q_f32(bpk + 0), bR = vld1q_f32(bpk + 4);
        const float* apk = Ap + (size_t)k*4;
        float32x4_t a = vld1q_f32(apk);
        float32x4_t a0 = dup_laneq0(a);
        float32x4_t a1 = dup_laneq1(a);
        float32x4_t a2 = dup_laneq2(a);
        float32x4_t a3 = dup_laneq3(a);

        accL[0] = vmlaq_f32(accL[0], a0, bL);  accR[0] = vmlaq_f32(accR[0], a0, bR);
        accL[1] = vmlaq_f32(accL[1], a1, bL);  accR[1] = vmlaq_f32(accR[1], a1, bR);
        accL[2] = vmlaq_f32(accL[2], a2, bL);  accR[2] = vmlaq_f32(accR[2], a2, bR);
        accL[3] = vmlaq_f32(accL[3], a3, bL);  accR[3] = vmlaq_f32(accR[3], a3, bR);
    }

    for (int r = 0; r < 4; ++r) {
        float* crow = Cbase + (size_t)r*N;
        vst1q_f32(crow + 0, vmulq_n_f32(accL[r], alpha));
        vst1q_f32(crow + 4, vmulq_n_f32(accR[r], alpha));
    }
}

// =============== 顶层：A+B pack + 8×8 双缓冲内核 ===============
void gemm_packAB_8x8_db(const float* __restrict A,
                         const float* __restrict B,
                         float* __restrict C,
                         int M, int N, int K,
                         float alpha)
{
    std::vector<float> Bpanel((size_t)K * 8);
    std::vector<float> Apanel8((size_t)K * 8);
    std::vector<float> Apanel4((size_t)K * 4);

    int j = 0;
    for (; j + 7 < N; j += 8) {
        // 打包 B 的 8 列
        pack_B_panel_8(B, N, K, j, 8, Bpanel.data());

        int i = 0;
        // 8 行块
        for (; i + 7 < M; i += 8) {
            pack_A_panel_8(A, M, K, i, 8, Apanel8.data());
            micro_kernel_8x8_packAB_db_k8(Apanel8.data(), Bpanel.data(),
                                           C + (size_t)i*N + j, N, K, alpha);
        }
        // 4 行尾
        if (i + 3 < M) {
            pack_A_panel_4(A, M, K, i, 4, Apanel4.data());
            micro_kernel_4x8_packAB_db_k8(Apanel4.data(), Bpanel.data(),
                                           C + (size_t)i*N + j, N, K, alpha);
            i += 4;
        }
        // 余下 <4 行：标量兜底
        for (; i < M; ++i) {
            float* crow = C + (size_t)i*N + j;
            const float* ai = A + (size_t)i*K;
            for (int c = 0; c < 8; ++c) {
                float sum = 0.f;
                const float* bp = Bpanel.data() + c; // 步长 8
                for (int k = 0; k < K; ++k) sum += ai[k] * bp[(size_t)k*8];
                crow[c] = alpha * sum;
            }
        }
    }

    // N 尾列（<8）：简单标量兜底（你的 N=80 不会走这里）
    if (j < N) {
        int cols = N - j;
        std::vector<float> Bptail((size_t)K * 8);
        pack_B_panel_8(B, N, K, j, cols, Bptail.data());

        int i = 0;
        for (; i + 7 < M; i += 8) {
            pack_A_panel_8(A, M, K, i, 8, Apanel8.data());
            // 逐列写回（cols <= 8）
            float tile[8*8];
            micro_kernel_8x8_packAB_db_k8(Apanel8.data(), Bptail.data(), tile, 8, K, alpha);
            for (int r = 0; r < 8; ++r) {
                float* crow = C + (size_t)(i+r)*N + j;
                for (int c = 0; c < cols; ++c) crow[c] = tile[r*8 + c];
            }
        }
        if (i + 3 < M) {
            pack_A_panel_4(A, M, K, i, 4, Apanel4.data());
            float tile[4*8];
            micro_kernel_4x8_packAB_db_k8(Apanel4.data(), Bptail.data(), tile, 8, K, alpha);
            for (int r = 0; r < 4; ++r) {
                float* crow = C + (size_t)(i+r)*N + j;
                for (int c = 0; c < cols; ++c) crow[c] = tile[r*8 + c];
            }
            i += 4;
        }
        for (; i < M; ++i) {
            float* crow = C + (size_t)i*N + j;
            const float* ai = A + (size_t)i*K;
            for (int c = 0; c < cols; ++c) {
                float sum = 0.f;
                const float* bp = Bptail.data() + c;
                for (int k = 0; k < K; ++k) sum += ai[k] * bp[(size_t)k*8];
                crow[c] = alpha * sum;
            }
        }
    }
}

// =============== 校验 & 计时 ===============
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
    const float alpha = 1.0f; // 可调

    std::vector<float> A((size_t)M*K), B((size_t)N*K), Cbase((size_t)M*N), Copt((size_t)M*N);

    // 随机初始化
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // Baseline（单次）
    auto t0 = std::chrono::high_resolution_clock::now();
    gemm_baseline(A.data(), B.data(), Cbase.data(), M, N, K, alpha);
    auto t1 = std::chrono::high_resolution_clock::now();
    double baseline_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // packAB + 8x8 双缓冲（多次平均）
    auto run_opt = [&](){
        gemm_packAB_8x8_db(A.data(), B.data(), Copt.data(), M, N, K, alpha);
    };
    double opt_ms = bench_avg_ms(run_opt, 2, 30);

    bool ok = check_result(Cbase, Copt, M, N, 1e-4f);

    std::cout << "Alpha = " << alpha << "\n";
    std::cout << "Baseline time (1 run): " << baseline_ms << " ms\n";
    std::cout << "PackAB+8x8 (avg)     : " << opt_ms      << " ms\n";
    std::cout << "Correctness: " << (ok ? "PASS" : "FAIL") << "\n";
    std::cout << "Target < 8ms? " << (opt_ms < 8.0 ? "YES" : "NO") << "\n";
    return 0;
}
