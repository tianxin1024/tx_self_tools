#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.7.5/bin:$PATH

export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub

curr_dir=$(pwd)


cd run/out
atc --singleop=test_data/config/matrixmulcust_op.json --soc_version=Ascend310P --output=op_models


cd $curr_dir
mkdir -p build/intermediates/host

cd build/intermediates/host
cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE

make

cd $curr_dir/run/out

./execute_custom_matrix_mul_cust_op




