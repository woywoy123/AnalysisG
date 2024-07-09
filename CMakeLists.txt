cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
set(SKBUILD_PROJECT_NAME AnalysisG)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX C)
include(FetchContent)

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
message(STATUS "INSTALLING EXTENSIONS WITH CUDA!")
string(REGEX REPLACE ".[0-9][0-9]|\\." "" CUDA_V ${CMAKE_CUDA_COMPILER_VERSION})
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CUDA_V}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HDF5)
find_package(Torch)
find_package(RapidJSON)
find_package(PythonInterp)
find_package(ROOT)

if (NOT ROOT_FOUND)
FetchContent_Declare(ROOT URL https://root.cern/download/root_v6.30.04.source.tar.gz FIND_PACKAGE_ARGS)
endif()

if (NOT RapidJSON_FOUND)
FetchContent_Declare(RapidJSON GIT_REPOSITORY https://github.com/Tencent/rapidjson.git)
FetchContent_MakeAvailable(RapidJSON)
find_package(RapidJSON REQUIRED)
endif()

if (NOT Torch_FOUND)
FetchContent_Declare(Torch URL https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip)
FetchContent_MakeAvailable(Torch)
list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")
endif()

if (NOT HDF5_FOUND)
set(HDF5_ENABLE_SZIP_SUPPORT OFF CACHE INTERNAL "")
set(HDF5_ENABLE_SZIP_ENCODING OFF CACHE INTERNAL "")
set(HDF5_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(HDF5_BUILD_CPP_LIB ON CACHE INTERNAL "")
set(HDF5_ENABLE_Z_LIB_SUPPORT OFF CACHE INTERNAL "")
set(HDF5_EXTERNALLY_CONFIGURED 1 CACHE INTERNAL "")
set(HDF5_NAMESPACE hdf5::)
FetchContent_Declare(HDF5
URL https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_3.tar.gz
URL_HASH SHA256=df5ee33c74d5efb59738075ef96f4201588e1f1eeb233f047ac7fd1072dee1f6 
DOWNLOAD_EXTRACT_TIMESTAMP FALSE
FIND_PACKAGE_ARGS)
FetchContent_MakeAvailable(HDF5)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(ROOT REQUIRED COMPONENTS RIO Tree Core)
find_package(HDF5 COMPONENTS C CXX REQUIRED)
find_package(Torch REQUIRED)

message(STATUS "ROOT: ${ROOT_INCLUDE_DIRS}")
include(${ROOT_USE_FILE})
include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
link_directories(${ROOT_LIBRARY_DIR})

set(HDF5_USE_CXX TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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
OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
)

file(COPY ${CMAKE_BINARY_DIR}/src/AnalysisG DESTINATION ${PYTHON_SITE_PACKAGES})
