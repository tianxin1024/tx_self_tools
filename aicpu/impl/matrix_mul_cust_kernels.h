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

#ifndef _AICPU_MATRIX_MUL_CUST_KERNELS_H_
#define _AICPU_MATRIX_MUL_CUST_KERNELS_H_

#include "cpu_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class MatrixMulCpuKernel : public CpuKernel {
public:
  ~MatrixMulCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;
  template <typename T> uint32_t MatrixMulCompute(CpuKernelContext &ctx);
  template <typename T>
  uint32_t MatrixMulComputeWithBlock(CpuKernelContext &ctx, uint32_t blockid,
                                     uint32_t blockdim);
};

} // namespace aicpu
#endif
