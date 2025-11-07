#*
# @file CMakeLists.txt
# @brief Build configuration for the pyc_physics module in the AnalysisG framework.
#/

#*
# @brief Specifies source files and compilation options for the pyc_physics library.
#
# - If CUDA is enabled, includes CUDA-specific files and options.
# - Otherwise, includes standard C++ files.
# - Links the library to pyc_utils and pyc_transform.
#/

if(CMAKE_ANALYSISG_CUDA)
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/physics.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/base.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/physics.cu
)

add_library(pyc_physics ${SOURCE_FILES})
target_compile_options(pyc_physics PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
set_property(TARGET pyc_physics PROPERTY CUDA_SEPARATE_COMPILATION ON)
else()

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/physics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/physics.cxx
)
add_library(pyc_physics ${SOURCE_FILES})
endif()

target_link_libraries(pyc_physics PUBLIC pyc_utils pyc_transform)
target_include_directories(pyc_physics PRIVATE include/physics)
target_include_directories(pyc_physics PUBLIC include)
target_compile_options(pyc_physics PRIVATE -fPIC)

