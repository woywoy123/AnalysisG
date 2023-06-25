cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/transform)
include_directories(${CMAKE_SOURCE_DIR}/physics)

add_subdirectory(transform)
add_subdirectory(physics)

