cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(transform VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)


add_subdirectory(cartesian-cuda)

add_library(TransformCuda SHARED transform-cuda.cxx)
target_link_libraries(TransformCuda PRIVATE cartesian-cuda Python::Python)
target_compile_options(TransformCuda PRIVATE $<$<COMPILE_LANGUAGE::CUDA>:-ccbin g++-9>)
