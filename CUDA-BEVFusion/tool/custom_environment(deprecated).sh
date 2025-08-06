#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Environment setup for CUDA-BEVFusion
# Sources this script and exports all required vars.

# Set TensorRT paths
export TensorRT_Lib=/usr/lib/x86_64-linux-gnu
export TensorRT_Inc=/usr/include/x86_64-linux-gnu
export TensorRT_Bin=/usr/src/tensorrt/bin

# Set CUDA paths
export CUDA_Lib=/usr/local/cuda/lib64
export CUDA_Inc=/usr/local/cuda/include
export CUDA_Bin=/usr/local/cuda-12.2/bin
export CUDA_HOME=/usr/local/cuda-12.2

# cuDNN
export CUDNN_Lib=/usr/lib/x86_64-linux-gnu

# spconv CUDA version tag
export SPCONV_CUDA_VERSION=12.8

# Model / precision flags
export DEBUG_MODEL=resnet50int8
export DEBUG_PRECISION=int8
export DEBUG_DATA=custom-example
export USE_Python=OFF

# Build directory
export BuildDirectory=$(pwd)/build

# Verify tools
if [ ! -x "${TensorRT_Bin}/trtexec" ]; then
    echo "Error: trtexec not found in $TensorRT_Bin" >&2
    exit 1
fi
if [ ! -x "${CUDA_Bin}/nvcc" ]; then
    echo "Error: nvcc not found in $CUDA_Bin" >&2
    exit 1
fi

# Python include / lib if needed
if [ "$USE_Python" == "ON" ]; then
    export Python_Inc=$(python3 -c "import sysconfig;print(sysconfig.get_path('include'))")
    export Python_Lib=$(python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
    export Python_Soname=$(python3 -c "import sysconfig,re;print(re.sub('.a','.so',sysconfig.get_config_var('LIBRARY')))" )
fi

# Prepend to PATH and LD_LIBRARY_PATH
export PATH="$TensorRT_Bin:$CUDA_Bin:$PATH"
export LD_LIBRARY_PATH="$TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:$BuildDirectory:$LD_LIBRARY_PATH"
export PYTHONPATH="$BuildDirectory:$PYTHONPATH"

echo "Environment configured."
