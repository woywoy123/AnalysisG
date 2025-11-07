#*
# @file CMakeLists.txt
# @brief Build configuration for the pyc_operators module in the AnalysisG framework.
#/

if(CMAKE_ANALYSISG_CUDA)
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/operators/operators.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/operators/base.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/operators.cu
)
add_library(pyc_operators ${SOURCE_FILES})
target_compile_options(pyc_operators PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
set_property(TARGET pyc_operators PROPERTY CUDA_SEPARATE_COMPILATION ON)
else()
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/operators/operators.h
    ${CMAKE_CURRENT_SOURCE_DIR}/operators.cxx
)
add_library(pyc_operators ${SOURCE_FILES})
endif()

#*
# @brief Specifies source files and compilation options for the pyc_operators library.
#
# - If CUDA is enabled, includes CUDA-specific files and options.
# - Otherwise, includes standard C++ files.
# - Links the library to pyc_utils.
#/

target_include_directories(pyc_operators PRIVATE include/operators)
target_include_directories(pyc_operators PUBLIC include)
target_link_libraries(pyc_operators PUBLIC pyc_utils)
target_compile_options(pyc_operators PRIVATE -fPIC)

