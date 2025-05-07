/**
 * @file CMakeLists.txt
 * @brief Build configuration for the pyc_nusol module in the AnalysisG framework.
 */

if (CMAKE_ANALYSISG_CUDA)
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/device.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/utils.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/base.cuh

    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/nu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/nunu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/intersection.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/matrix.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/base.cu
)

add_library(pyc_nusol-base ${SOURCE_FILES})
target_include_directories(pyc_nusol-base PRIVATE include/nusol)
target_include_directories(pyc_nusol-base PUBLIC include)
target_link_libraries(pyc_nusol-base PUBLIC pyc_utils pyc_physics pyc_operators)
set_property(TARGET pyc_nusol-base PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(pyc_nusol-base PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_compile_options(pyc_nusol-base PRIVATE -fPIC)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/nusol.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/nusol.cu
)

add_library(pyc_nusol ${SOURCE_FILES})
target_link_libraries(pyc_nusol PUBLIC pyc_nusol-base)
set_property(TARGET pyc_nusol PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(pyc_nusol PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
else()
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/nusol.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/nusol.cxx
)
add_library(pyc_nusol ${SOURCE_FILES})
endif()

/**
 * @brief Specifies source files and compilation options for the pyc_nusol library.
 *
 * - If CUDA is enabled, includes CUDA-specific files and options.
 * - Otherwise, includes standard C++ files.
 * - Links the library to pyc_utils, pyc_physics, pyc_operators, and pyc_transform.
 */
target_include_directories(pyc_nusol PRIVATE include/nusol)
target_include_directories(pyc_nusol PUBLIC include)
target_link_libraries(pyc_nusol PUBLIC pyc_utils pyc_physics pyc_operators pyc_transform)
target_compile_options(pyc_nusol PRIVATE -fPIC)

