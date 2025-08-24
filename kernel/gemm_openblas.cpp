#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "/opt/homebrew/opt/openblas/include/cblas.h"

#define MIN_DIM 256
#define MAX_DIM 2048
#define STEP 256

// 简单的参考GEMM实现（用于验证）
void gemm_naive(int m, int n, int k, double alpha, double* A, double* B, double beta, double* C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

// 性能测试函数
void benchmark_gemm() {
    printf("开始性能测试...\n");
    printf("维度\tOpenBLAS时间(s)\t性能(GFLOPS)\n");
    printf("----------------------------------------\n");
    
        int m = 900, k = 512, n = 80;
        
        // 分配内存
        double* A = (double*)malloc(m * k * sizeof(double));
        double* B = (double*)malloc(k * n * sizeof(double));
        double* C = (double*)malloc(m * n * sizeof(double));
        
        // 初始化矩阵
        for (int i = 0; i < m * k; i++) A[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < k * n; i++) B[i] = (double)rand() / RAND_MAX;
        for (int i = 0; i < m * n; i++) C[i] = (double)rand() / RAND_MAX;
        
        double alpha = 1.0;
        double beta = 0.0;
        
        // 预热缓存
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);
        
        // 正式计时
        clock_t start = clock();
        int iterations = 3;  // 多次运行取平均
        for (int i = 0; i < iterations; i++) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, alpha, A, k, B, n, beta, C, n);
        }
        clock_t end = clock();
        
        double time_used = (double)(end - start) / CLOCKS_PER_SEC / iterations;
        double flops = 2.0 * m * n * k;
        double gflops = flops / (time_used * 1e9);
        
        printf("900 x 80\t%.4f\t\t%.2f\n", time_used, gflops);
        
        free(A);
        free(B);
        free(C);
}

int main() {
    srand(time(NULL));
    
    // 运行性能测试
    benchmark_gemm();
    
    return 0;
}
