cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(SKBUILD_PROJECT_NAME AnalysisG)
project(${SKBUILD_PROJECT_NAME}  VERSION 5.0 LANGUAGES CXX)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

find_package(ROOT   COMPONENTS RIO Tree Core REQUIRED)
find_package(HDF5   REQUIRED COMPONENTS CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

get_target_property(ROOT_CXX_STANDARD ROOT::Core INTERFACE_COMPILE_FEATURES)
string(REGEX MATCH "[0-9]+" ROOT_CXX_STANDARD "${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES "all")
#set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dimt=ON")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")

set(CMAKE_ANALYSISG_CUDA ON)
include(FetchContent)
include(CheckLanguage)
set(USE_SABI 1)

set(RAPIDJSON_BUILD_DOC OFF)
set(RAPIDJSON_BUILD_CXX11 OFF)
set(RAPIDJSON_BUILD_CXX17 OFF)
set(RAPIDJSON_BUILD_TESTS OFF)
set(RAPIDJSON_BUILD_EXAMPLES OFF)
set(RAPIDJSON_BUILD_THIRDPARTY_GTEST OFF)
FetchContent_Declare(rapidjson
GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
GIT_TAG origin/master
FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(rapidjson)
set(RAPIDJSON_INCLUDE_DIR ${CMAKE_BINARY_DIR}/_deps/rapidjson-src/include)

set(USE_CUDSS ON)
set(USE_CUDNN ON)
set(USE_CUFILE ON)
if(CMAKE_ANALYSISG_CUDA)
check_language(CUDA)
enable_language(CUDA)
#find_package(CUDA REQUIRED)
message(STATUS "INSTALLING EXTENSIONS WITH CUDA!")
string(REGEX REPLACE ".[0-9][0-9]|\\." "" CUDA_V ${CMAKE_CUDA_COMPILER_VERSION})
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CUDA_V}")
FetchContent_Declare(torch URL "https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.1%2Bcu126.zip")
else()
FetchContent_Declare(torch URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.1%2Bcpu.zip")
endif()
FetchContent_MakeAvailable(torch)

execute_process(
COMMAND python -c "import sysconfig; print(sysconfig.get_paths()['platlib'])" 
OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${torch_SOURCE_DIR})

find_package(Torch REQUIRED CONFIG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "CAFFE2_USE_CUDNN is: ${CAFFE2_USE_CUDNN}")
list(APPEND CMAKE_PREFIX_PATH "${ROOT_SOURCE_DIR}")
include_directories(${ROOT_INCLUDE_DIRS})

message(STATUS      "ROOT: ${ROOT_LIBRARIES}")
message(STATUS      "HDF5: ${HDF5_CXX_INCLUDE_DIRS}")
message(STATUS "RAPIDJSON: ${RAPIDJSON_INCLUDE_DIR}")
message(STATUS     "TORCH: ${TORCH_LIBRARIES}")

add_subdirectory(src/AnalysisG)
file(INSTALL ${CMAKE_BINARY_DIR}/src/AnalysisG DESTINATION ${PYTHON_SITE_PACKAGES})
