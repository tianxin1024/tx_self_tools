"""
* @file generate_data.py
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
import numpy as np

# 设置随机种子以便结果可重现（可选）
np.random.seed(42)

# 定义矩阵维度
M, N, K = 900, 80, 512

# 生成随机矩阵 A 和 B，使用 float32 类型
A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

# 矩阵乘法
C = A @ B  # 或者使用 np.matmul(a, b)

# 输出结果
print("Matrix A:")
print(A)
print(A.shape)
print("\nMatrix B:")
print(B)
print(B.shape)
print("\nMatrix C (A * B):")
print(C)
print(C.shape)

A.tofile('input_0.bin')
B.tofile('input_1.bin')


