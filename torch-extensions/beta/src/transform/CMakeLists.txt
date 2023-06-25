cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(transform VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

add_subdirectory(cartesian-floats)
add_subdirectory(polar-floats)
add_library(TransformFloats SHARED transform-floats.cxx)
target_link_libraries(TransformFloats PRIVATE ${TORCH_LIBRARIES} Python::Python cartesian-floats polar-floats)

add_subdirectory(cartesian-tensors)
add_subdirectory(polar-tensors)
add_library(TransformTensors SHARED transform-tensors.cxx)
target_link_libraries(TransformTensors PRIVATE ${TORCH_LIBRARIES} Python::Python cartesian-tensors polar-tensors)

add_subdirectory(cartesian-cuda)
add_subdirectory(polar-cuda)
add_library(TransformCuda SHARED transform-cuda.cxx)
target_link_libraries(TransformCuda PRIVATE cartesian-cuda polar-cuda Python::Python)
target_compile_options(TransformCuda PRIVATE $<$<COMPILE_LANGUAGE::CUDA>:-ccbin g++-10>)

