#!/bin/bash

# g++ -O2 -ffast-math -march=armv8-a+simd gemm_v2.cpp -o gemm

# ./gemm
#
#
# g++ -O2 -std=c++11 -ffast-math -march=armv8-a+simd gemm_v4.cpp -o gemm
# ./gemm


# gemm_pack8x8
# g++ -O3 -ffast-math -march=armv8-a+simd -std=c++14 gemm_pack8x8.cpp -o gemm
# # 或者直接 -mcpu=native 视芯片而定
# ./gemm

# gemm_pack8x8_db_k8
# g++ -O3 -ffast-math -march=armv8-a+simd -std=c++14 gemm_pack_8x8_db_k8.cpp -o gemm
# # 或者直接 -mcpu=native 视芯片而定
# ./gemm


# gemm_packAB_8x8_db
g++ -Ofast -ffast-math -funroll-loops -fomit-frame-pointer \
    -march=armv8-a+simd -mtune=native -std=c++14 \
    gemm_packAB_8x8_db.cpp -o gemm
# 如果支持 LTO：
# g++ ... -flto
# 或者直接 -mcpu=native 视芯片而定
./gemm

