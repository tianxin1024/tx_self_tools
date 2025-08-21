#!/bin/bash

# g++ -O2 -ffast-math -march=armv8-a+simd gemm_v2.cpp -o gemm

# ./gemm
#
#
g++ -O2 -std=c++11 -ffast-math -march=armv8-a+simd gemm_v4.cpp -o gemm

./gemm
