set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/operators/operators.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/operators/base.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/operators.cu
)

add_library(cuoperators ${SOURCE_FILES})
target_include_directories(cuoperators PRIVATE include/operators)
target_include_directories(cuoperators PUBLIC include)
target_compile_options(cuoperators PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_link_libraries(cuoperators PUBLIC cutils cuatomic)
set_property(TARGET cuoperators PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cuoperators PRIVATE -fPIC)


