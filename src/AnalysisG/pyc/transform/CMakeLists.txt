#*
# @file CMakeLists.txt
# @brief Build configuration for the pyc_transform module in the AnalysisG framework.
#/

#*
# @brief Specifies source files and compilation options for the pyc_transform library.
#
# - If CUDA is enabled, includes CUDA-specific files and options.
# - Otherwise, includes standard C++ files.
# - Sets include directories and links the library to pyc_utils.
#/

if(CMAKE_ANALYSISG_CUDA)
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/transform/transform.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/transform/base.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/transform.cu
)

add_library(pyc_transform ${SOURCE_FILES})
target_compile_options(pyc_transform PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
set_property(TARGET pyc_transform PROPERTY CUDA_SEPARATE_COMPILATION ON)
else()
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/transform/transform.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transform.cxx
)
add_library(pyc_transform ${SOURCE_FILES})
endif()
target_include_directories(pyc_transform PRIVATE include/transform)
target_include_directories(pyc_transform PUBLIC include)
target_link_libraries(pyc_transform PUBLIC pyc_utils)
target_compile_options(pyc_transform PRIVATE -fPIC)

