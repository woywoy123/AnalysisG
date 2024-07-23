cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
set(SKBUILD_PROJECT_NAME AnalysisG)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX C)

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

find_package(Torch)
find_package(PythonInterp)
find_package(RapidJSON)
find_package(HDF5)
find_package(ROOT)

if (NOT rapidjson_FOUND)
set(RAPIDJSON_BUILD_EXAMPLES OFF)
set(RAPIDJSON_BUILD_TESTS OFF)
set(RAPIDJSON_BUILD_DOC OFF)
set(RAPIDJSON_BUILD_CXX11 OFF)
set(RAPIDJSON_BUILD_CXX17 ON)
FetchContent_Declare(rapidjson
GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
GIT_TAG v1.1.0)
FetchContent_MakeAvailable(rapidjson)
endif()

#FetchContent_Declare(ROOT
#URL "https://root.cern/download/root_v6.32.02.Linux-almalinux9.4-x86_64-gcc11.4.tar.gz"
#FIND_PACKAGE_ARGS)
#FetchContent_MakeAvailable(ROOT)
#list(APPEND CMAKE_PREFIX_PATH "${ROOT_SOURCE_DIR}")

if (NOT Torch_FOUND)
FetchContent_Declare(Torch 
URL "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip")
FetchContent_MakeAvailable(Torch)
list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")
endif()

if (NOT hdf5_FOUND)
set(HDF5_USE_CXX ON)
set(BUILD_TESTING OFF)
set(HDF5_BUILD_EXAMPLES OFF)
set(HDF5_BUILD_CPP_LIB ON)
set(HDF5_EXTERNALLY_CONFIGURED ON)
FetchContent_Declare(hdf5
GIT_REPOSITORY "https://github.com/HDFGroup/hdf5.git"
GIT_TAG hdf5_1.14.4.3)
FetchContent_MakeAvailable(hdf5)
endif()

include(${ROOT_USE_FILE})
link_directories(${ROOT_LIBRARY_DIR})
message(STATUS "ROOT: ${ROOT_INCLUDE_DIRS}")
include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
list(APPEND CMAKE_PREFIX_PATH "${ROOT_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(ROOT REQUIRED COMPONENTS RIO Tree Core)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(HDF5 COMPONENTS CXX REQUIRED)
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

