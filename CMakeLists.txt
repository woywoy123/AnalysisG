cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
set(SKBUILD_PROJECT_NAME AnalysisG)
project(${SKBUILD_PROJECT_NAME} VERSION 5.0 LANGUAGES CXX C)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)

include(FetchContent)
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
enable_language(CUDA)
message(STATUS "INSTALLING EXTENSIONS WITH CUDA!")
string(REGEX REPLACE ".[0-9][0-9]|\\." "" CUDA_V ${CMAKE_CUDA_COMPILER_VERSION})
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CUDA_V}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERSION 3.20)

find_package(Torch)
find_package(PythonInterp)
find_package(HDF5)

if (NOT rapidjson_FOUND)
set(RAPIDJSON_BUILD_EXAMPLES OFF)
set(RAPIDJSON_BUILD_TESTS OFF)
set(RAPIDJSON_BUILD_DOC OFF)
set(RAPIDJSON_BUILD_CXX11 ON)
set(RAPIDJSON_BUILD_CXX17 ON)
set(RAPIDJSON_BUILD_THIRDPARTY_GTEST ON)
FetchContent_Declare(rapidjson
GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
GIT_TAG origin/master
FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(rapidjson)
set(RAPIDJSON_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/rapidjson-src/include)
message(STATUS "RAPIDJSON: ${RAPIDJSON_INCLUDE_DIR}")
endif()

#find_package(ROOT)
#FetchContent_Declare(root
#GIT_REPOSITORY "https://github.com/root-project/root.git"
#GIT_TAG "v6-32-02")
#FetchContent_MakeAvailable(root)
#
#list(APPEND CMAKE_PREFIX_PATH "${ROOT_SOURCE_DIR}")
#include(${ROOT_USE_FILE})
#link_directories(${ROOT_LIBRARY_DIR})
#message(STATUS "ROOT: ${ROOT_INCLUDE_DIRS}")
#include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})


if (NOT Torch_FOUND)
FetchContent_Declare(Torch 
URL "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip")
FetchContent_MakeAvailable(Torch)
list(APPEND CMAKE_PREFIX_PATH ${torch_SOURCE_DIR})
endif()

#if (NOT HDF5_FOUND)
#FetchContent_Declare(HDF5
#GIT_REPOSITORY "https://github.com/HDFGroup/hdf5.git"
#GIT_TAG origin/hdf5_1_14_4
#OVERRIDE_FIND_PACKAGE)
#option(BUILD_TESTING "" OFF)
#option(HDF5_BUILD_SHARED_LIBS "" ON)
#option(HDF5_BUILD_CPP_LIB "" ON)
#option(HDF5_ENABLE_SZIP_SUPPORT "" OFF)
#option(HDF5_ENABLE_SZIP_ENCODING "" OFF)
#option(HDF5_BUILD_EXAMPLES "" OFF)
#option(HDF5_BUILD_CPP_LIB "" ON)
#option(HDF5_ENABLE_Z_LIB_SUPPORT "" ON)
#FetchContent_MakeAvailable(HDF5)
#message(STATUS "HDF5: ${HDF5_LIBRARIES}")
#endif()

list(APPEND CMAKE_PREFIX_PATH "${ROOT_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(ROOT REQUIRED COMPONENTS RIO Tree Core)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Torch REQUIRED)

add_subdirectory(src/AnalysisG/pyc)
add_subdirectory(src/AnalysisG/modules)
add_subdirectory(src/AnalysisG/generators)
add_subdirectory(src/AnalysisG/events)
add_subdirectory(src/AnalysisG/selections)
add_subdirectory(src/AnalysisG/graphs)
add_subdirectory(src/AnalysisG/models)
add_subdirectory(src/AnalysisG/core)

execute_process(
COMMAND python -c "import sysconfig; print(sysconfig.get_paths()['platlib'])" 
OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
file(INSTALL ${CMAKE_BINARY_DIR}/src/AnalysisG DESTINATION ${PYTHON_SITE_PACKAGES})

