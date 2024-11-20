set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/atomic/cuatomic.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/cuatomic.cu
)

add_library(cuatomic ${SOURCE_FILES})
target_include_directories(cuatomic PRIVATE include/atomic)
target_include_directories(cuatomic PUBLIC include)
target_compile_options(cuatomic PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_link_libraries(cuatomic PUBLIC ${TORCH_LIBRARIES})
set_property(TARGET cuatomic PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cuatomic PRIVATE -fPIC)
