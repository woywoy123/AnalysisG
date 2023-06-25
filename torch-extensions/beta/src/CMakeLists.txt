cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(PyExt VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

add_subdirectory(transform)
