cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(SKBUILD_PROJECT_NAME AnalysisG)
project(${SKBUILD_PROJECT_NAME}  VERSION 5.0 LANGUAGES CXX)

# ABI is the pre-cxx11 version that pytorch distributes via wheels...
#set(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "native")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dimt=ON")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_ANALYSISG_CUDA ON)
include(FetchContent)
include(CheckLanguage)

set(RAPIDJSON_BUILD_THIRDPARTY_GTEST OFF)
set(RAPIDJSON_BUILD_EXAMPLES OFF)
set(RAPIDJSON_BUILD_TESTS OFF)
set(RAPIDJSON_BUILD_DOC OFF)
set(RAPIDJSON_BUILD_CXX11 OFF)
set(RAPIDJSON_BUILD_CXX17 ON)
FetchContent_Declare(rapidjson
GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
GIT_TAG origin/master
FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(rapidjson)
set(RAPIDJSON_INCLUDE_DIR ${CMAKE_BINARY_DIR}/_deps/rapidjson-src/include)

if(CMAKE_ANALYSISG_CUDA)
check_language(CUDA)
enable_language(CUDA)
message(STATUS "INSTALLING EXTENSIONS WITH CUDA!")
string(REGEX REPLACE ".[0-9][0-9]|\\." "" CUDA_V ${CMAKE_CUDA_COMPILER_VERSION})
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CUDA_V}")
FetchContent_Declare(torch 
URL "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu121.zip")
else()
FetchContent_Declare(torch 
URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip")
endif()
FetchContent_MakeAvailable(torch)

execute_process(
COMMAND python -c "import sysconfig; print(sysconfig.get_paths()['platlib'])" 
OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

set(BLAS MKL)
set(USE_TBB ON)
set(MKL_TBB ON)
set(USE_MKL ON)
set(USE_MKLDNN ON)

set(USE_MPI OFF)
set(USE_OPENMP OFF)
set(BUILD_BINARY ON)

set(MKL_THREADING TBB)
set(PARALLEL_BACKEND TBB)
set(MKLDNN_THREADING TBB)
set(MKLDNN_CPU_RUNTIME TBB)
set(USE_EIGEN_THREADPOOL ON)

set(USE_NNPACK ON)
set(USE_NCCL ON)
set(USE_CUDA ON)
set(USE_CUDNN ON)
set(CAFFE2_USE_CUDNN TRUE)
set(BUILD_CAFFE2_OPS ON)

list(APPEND CMAKE_PREFIX_PATH ${torch_SOURCE_DIR})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "CAFFE2_USE_CUDNN is: ${CAFFE2_USE_CUDNN}")

#FetchContent_Declare(HDF5
#GIT_REPOSITORY "https://github.com/HDFGroup/hdf5.git"
#GIT_TAG origin/hdf5_1_14_5
#FIND_PACKAGE_ARGS)
#set(BUILD_TESTING "" OFF)
#set(HDF5_BUILD_CPP_LIB "" ON)
#set(HDF5_BUILD_SHARED_LIBS "" ON)
#set(HDF5_BUILD_EXAMPLES "" OFF)
#set(HDF5_ENABLE_SZIP_SUPPORT "" OFF)
#set(HDF5_ENABLE_SZIP_ENCODING "" OFF)
#set(HDF5_ENABLE_Z_LIB_SUPPORT "" OFF)
#FetchContent_MakeAvailable(HDF5)
#list(APPEND CMAKE_PREFIX_PATH ${HDF5_SOURCE_DIR})

#find_package(HDF5   NAMES hdf5 COMPONENTS cpp-shared REQUIRED)
find_package(HDF5   REQUIRED COMPONENTS CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(ROOT   COMPONENTS RIO Tree Core REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "${ROOT_SOURCE_DIR}")
include_directories(${ROOT_INCLUDE_DIRS})

message(STATUS      "ROOT: ${ROOT_LIBRARIES}")
message(STATUS      "HDF5: ${HDF5_CXX_INCLUDE_DIRS}")
message(STATUS "RAPIDJSON: ${RAPIDJSON_INCLUDE_DIR}")
message(STATUS     "TORCH: ${TORCH_LIBRARIES}")

add_subdirectory(src/AnalysisG)
#file(INSTALL ${CMAKE_BINARY_DIR}/src/AnalysisG DESTINATION ${PYTHON_SITE_PACKAGES})
