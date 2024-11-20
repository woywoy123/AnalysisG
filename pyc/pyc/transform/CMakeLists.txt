set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/transform/transform.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/transform/base.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/transform.cu
)

add_library(cutransform ${SOURCE_FILES})
target_include_directories(cutransform PRIVATE include/transform)
target_include_directories(cutransform PUBLIC include)
target_compile_options(cutransform PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_link_libraries(cutransform PUBLIC cutils cuatomic)
set_property(TARGET cutransform PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cutransform PRIVATE -fPIC)


