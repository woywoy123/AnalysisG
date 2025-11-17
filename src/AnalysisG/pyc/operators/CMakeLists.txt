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
target_link_libraries(pyc_operators PUBLIC pyc_transform)
target_link_libraries(pyc_operators PUBLIC pyc_physics)
endif()
target_include_directories(pyc_operators PRIVATE include/operators)
target_include_directories(pyc_operators PUBLIC include)
target_link_libraries(pyc_operators PUBLIC pyc_utils)
target_compile_options(pyc_operators PRIVATE -fPIC)

