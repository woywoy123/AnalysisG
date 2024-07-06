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

#if (DEFINED ENV{ATLAS_LOCAL_ROOT_BASE})
#    message("Setup up ATLAS with 'lsetup hdf5 1.8.17-sl7'!")
#endif()
#
#if (DEFINED ENV{ALRB_HDF5_PATH})
#    set(ENV{PATH} $ENV{ALRB_HDF5_PATH}:$ENV{PATH})
#endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonInterp)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(HDF5 COMPONENTS C CXX REQUIRED)
find_package(ROOT REQUIRED COMPONENTS RIO Tree Core)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(HDF5_USE_CXX TRUE)
include(${ROOT_USE_FILE})
include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
link_directories(${ROOT_LIBRARY_DIR})
message(STATUS "ROOT: ${ROOT_INCLUDE_DIRS}")

find_package(RapidJSON)
if (NOT RapidJSON_FOUND)
FetchContent_Declare(RapidJSON GIT_REPOSITORY https://github.com/Tencent/rapidjson.git)
FetchContent_MakeAvailable(RapidJSON)
find_package(RapidJSON REQUIRED)
endif()

find_package(Torch)
if (NOT Torch_FOUND)
FetchContent_Declare(Torch URL https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip)
FetchContent_MakeAvailable(Torch)
list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")
find_package(Torch REQUIRED)
endif()
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
