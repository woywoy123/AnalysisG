cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(physics VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(physics-cuda)
add_library(PhysicsPolar SHARED physics-cuda.cxx)
target_link_libraries(PhysicsPolar PUBLIC ${TORCH_LIBRARIES} Python::Python transform physics-cuda)
