/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "matrix_mul_cust_kernels.h"
#include <cmath>
#include <string>

#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"

#include <arm_neon.h>
#include <chrono>

namespace {
const char *MATRIX_MUL_CUST = "MatrixMulCust";
const uint32_t kFirstInputIndex = 0;
const uint32_t kSecondInputIndex = 1;
const uint32_t kFirstOutputIndex = 0;
const uint32_t SUCCESS = 0;
const uint32_t PARAM_INVAILD = 1;
const uint32_t ERROR = 2;
} // namespace

namespace aicpu {
uint32_t MatrixMulCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *input0 = ctx.Input(kFirstInputIndex);
  Tensor *input1 = ctx.Input(kSecondInputIndex);
  std::string opType = ctx.GetOpType();
  if (input0->GetDataSize() == 0 || input1->GetDataSize() == 0) {
    return SUCCESS;
  }

  auto data_type = static_cast<DataType>(input0->GetDataType());
  switch (data_type) {
  case DT_FLOAT:
    return MatrixMulCompute<float>(ctx);
  case DT_INT32:
    return MatrixMulCompute<int32_t>(ctx);
  case DT_INT64:
    return MatrixMulCompute<int64_t>(ctx);
  default:
    return PARAM_INVAILD;
  }
  return SUCCESS;
}

template <typename T>
uint32_t MatrixMulCpuKernel::MatrixMulCompute(CpuKernelContext &ctx) {
  // Get blockid and blockdim
  uint32_t blockid;
  uint32_t blockdim;
  AttrValue *block_id_ptr = ctx.GetAttr("block_id");
  AttrValue *block_dim_ptr = ctx.GetAttr("block_num");

  // check block_id and block_num
  if (block_id_ptr == nullptr || block_dim_ptr == nullptr) {
    blockid = 0;
    blockdim = 1;
  } else {
    blockid = block_id_ptr->GetInt();
    blockdim = block_dim_ptr->GetInt();
  }
  if (blockid >= blockdim || blockid < 0) {
    blockid = 0;
    blockdim = 1;
  }

  return MatrixMulComputeWithBlock<T>(ctx, blockid, blockdim);
}

void gemm_baseline(const float *A, const float *B, float *C, int M, int N,
                   int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[j * K + k]; // B 是 N×K row-major
      }
      C[i * N + j] = sum;
    }
  }
}

// ===================== Optimized NEON (之前的核心) =====================
#define A_AT(A, M, K, i, k) (A[(size_t)(i) * (K) + (k)])
#define B_AT(B, N, K, j, k) (B[(size_t)(j) * (K) + (k)])
#define C_AT(C, M, N, i, j) (C[(size_t)(i) * (N) + (j)])

static inline void micro_kernel_8x4_f32(const float *__restrict A,
                                        const float *__restrict B,
                                        float *__restrict C, int M, int N,
                                        int K, int i, int j) {
  float32x4_t acc[8][4];
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 4; ++c)
      acc[r][c] = vdupq_n_f32(0.0f);

  const float *bptr[4] = {
      &B_AT(B, N, K, j + 0, 0),
      &B_AT(B, N, K, j + 1, 0),
      &B_AT(B, N, K, j + 2, 0),
      &B_AT(B, N, K, j + 3, 0),
  };

  const float *aptr[8];
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

static inline void kernel_scalar_f32(const float *__restrict A,
                                     const float *__restrict B,
                                     float *__restrict C, int M, int N, int K,
                                     int i0, int i1, int j0, int j1) {
  for (int i = i0; i < i1; ++i) {
    for (int j = j0; j < j1; ++j) {
      const float *a = &A_AT(A, M, K, i, 0);
      const float *b = &B_AT(B, N, K, j, 0);
      float sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += a[k] * b[k];
      }
      C_AT(C, M, N, i, j) = sum;
    }
  }
}

void gemm_at_bt_f32_neon(const float *__restrict A, const float *__restrict B,
                         float *__restrict C, int M, int N, int K) {
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

template <typename T>
uint32_t MatrixMulCpuKernel::MatrixMulComputeWithBlock(CpuKernelContext &ctx,
                                                       uint32_t blockid,
                                                       uint32_t blockdim) {
  Tensor *input0 = ctx.Input(kFirstInputIndex);
  Tensor *input1 = ctx.Input(kSecondInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);

  float *A = reinterpret_cast<float *>(input0->GetData());
  if (A == nullptr) {
    return PARAM_INVAILD;
  }
  float *B = reinterpret_cast<float *>(input1->GetData());
  if (B == nullptr) {
    return PARAM_INVAILD;
  }
  float *C = reinterpret_cast<float *>(output->GetData());
  if (C == nullptr) {
    return PARAM_INVAILD;
  }

  // caculate per unit if blockdimByIndex = -1
  int64_t total = input0->NumElements();
  int64_t startpos = 0;
  int64_t len = total;
  if (blockdim != 1) {
    uint32_t per_unit = std::ceil(total / blockdim);
    startpos = blockid * per_unit;
    len =
        blockid < blockdim - 1 ? per_unit : (total - per_unit * (blockdim - 1));
  }

  // for (int i = startpos; i < startpos + len; i++) {
  //   y[i] = x0[i] + x1[i];
  // }

  int M = 900, N = 80, K = 512;

  auto time_start = std::chrono::steady_clock::now();
  gemm_baseline(A, B, C, M, N, K);
  auto time_end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> time_used = time_end - time_start;

  C[0] = time_used.count();

  gemm_at_bt_f32_neon(A, B, C, M, N, K);

  return SUCCESS;
}

REGISTER_CPU_KERNEL(MATRIX_MUL_CUST, MatrixMulCpuKernel);

} // namespace aicpu
