cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(pyc VERSION 5.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "native")

include(CheckLanguage)
include(FetchContent)

check_language(CUDA)
enable_language(CUDA)

message(STATUS "INSTALLING EXTENSIONS WITH CUDA!")
string(REGEX REPLACE ".[0-9][0-9]|\\." "" CUDA_V ${CMAKE_CUDA_COMPILER_VERSION})
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CUDA_V}")

FetchContent_Declare(torch URL "https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.5.1%2Bcu121.zip")
#"https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu121.zip")
FetchContent_MakeAvailable(torch)

list(APPEND CMAKE_PREFIX_PATH ${torch_SOURCE_DIR})
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(TORCH_USE_CUDA_DSA ON)
message(STATUS "CAFFE2_USE_CUDNN is: ${CAFFE2_USE_CUDNN}")
message(STATUS "TORCH: ${TORCH_LIBRARIES}")

add_subdirectory(pyc)
