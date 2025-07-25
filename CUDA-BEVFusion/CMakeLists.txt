# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.14)
project(bevfusion LANGUAGES C CXX)

# Dynamically fetch the stb library headers from GitHub
include(FetchContent)
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG master
)
FetchContent_MakeAvailable(stb)

set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)

set(spconv_include ../libraries/3DSparseConvolution/libspconv/include)
set(spconv_lib ../libraries/3DSparseConvolution/libspconv/lib/${arch}_cuda$ENV{SPCONV_CUDA_VERSION})
message(STATUS "spconv_lib = ${spconv_lib}")

if("$ENV{SPCONV_CUDA_VERSION}" EQUAL "11.4")
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_FLAGS_RELEASE "-Wextra -Wall -O3 -DENABLE_TEXT_BACKEND_STB")
  set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -DENABLE_TEXT_BACKEND_STB")
else()
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS_RELEASE "-Wextra -Wall -O3 -DENABLE_TEXT_BACKEND_STB")
  set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -DENABLE_TEXT_BACKEND_STB")
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_$ENV{CUDASM},code=compute_$ENV{CUDASM}
)

# Global include paths
include_directories(
  ${Protobuf_INCLUDE_DIRS}
  $ENV{CUDA_Inc}
  $ENV{TensorRT_Inc}
  $ENV{Python_Inc}
  ${stb_SOURCE_DIR}          # stb headers (stb_image, stb_image_write, etc.)
  ${CMAKE_CURRENT_SOURCE_DIR}/include  # project include dir
  ../dependencies/dlpack/include
  ../dependencies/pybind11/include
  ../libraries/cuOSD/src
  ${spconv_include}
  src/common
  src/
)

link_directories(
  $ENV{CUDA_Lib}
  $ENV{TensorRT_Lib}
  ${spconv_lib}
  build
  $ENV{Python_Lib}
)

file(GLOB_RECURSE CORE_FILES
  src/bevfusion/*.cu
  src/bevfusion/*.cpp
  src/onnx/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES
  ../libraries/cuOSD/src/*.cpp
  ../libraries/cuOSD/src/*.cu
  src/common/visualize.cu
)

# Build the core shared library
cuda_add_library(bevfusion_core SHARED ${CORE_FILES})
target_link_libraries(bevfusion_core
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
)

# Build executables
foreach(exec_name IN ITEMS bevfusion custom_bevfusion)
  cuda_add_executable(${exec_name} src/custom_main.cpp ${VISUALIZE_FILES})
  target_link_libraries(${exec_name}
    bevfusion_core
    libnvinfer.so
    libspconv.so
    ${Protobuf_LIBRARIES}
    libnvinfer_plugin.so
  )
endforeach()

# Custom layernorm plugin
cuda_add_library(custom_layernorm SHARED src/plugins/custom_layernorm.cu)
target_link_libraries(custom_layernorm libnvinfer.so libnvinfer_plugin.so)

# Optional Python binding
if(CMAKE_BUILD_PYTHON STREQUAL "ON")
  cuda_add_library(pybev SHARED src/python.cpp)
  target_link_libraries(pybev
    bevfusion_core
    libnvinfer.so
    libspconv.so
    ${Protobuf_LIBRARIES}
    libnvinfer_plugin.so
    $ENV{Python_Soname}
  )
endif()
