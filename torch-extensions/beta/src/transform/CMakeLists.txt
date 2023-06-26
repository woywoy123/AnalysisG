cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(transform VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#add_subdirectory(cartesian-floats)
#add_subdirectory(polar-floats)
#add_library(TransformFloats SHARED transform-floats.cxx)
#target_link_libraries(TransformFloats PRIVATE ${TORCH_LIBRARIES} Python::Python cartesian-floats polar-floats)
#
#add_subdirectory(cartesian-tensors)
#add_subdirectory(polar-tensors)
#add_library(TransformTensors SHARED transform-tensors.cxx)
#target_link_libraries(TransformTensors PRIVATE ${TORCH_LIBRARIES} Python::Python cartesian-tensors polar-tensors)
#
add_subdirectory(cartesian-cuda)
add_subdirectory(polar-cuda)
#add_library(TransformCuda SHARED transform-cuda.cxx)
#target_link_libraries(TransformCuda PRIVATE ${TORCH_LIBRARIES} Python::Python cartesian-cuda polar-cuda)
#target_link_libraries(transform_cu PRIVATE ${TORCH_LIBRARIES} Python::Python cartesian-cuda polar-cuda)

add_library(transform_cu "")
#target_sources(transform_cu PUBLIC cartesian-cuda polar-cuda)
#            ${CMAKE_CURRENT_LIST_DIR}/cartesian-cuda/cartesian.cu
#            ${CMAKE_CURRENT_LIST_DIR}/cartesian-cuda/cartesian.h
#            ${CMAKE_CURRENT_LIST_DIR}/cartesian-cuda/kernel.cu
#            ${CMAKE_CURRENT_LIST_DIR}/cartesian-cuda/torch.cu
#)
target_link_libraries(transform_cu PUBLIC ${TORCH_LIBRARIES} Python::Python cartesian-cuda polar-cuda)
target_include_directories(transform_cu PUBLIC ${CMAKE_CURRENT_LIST_DIR})

