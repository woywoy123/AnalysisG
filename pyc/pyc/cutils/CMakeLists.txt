set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cutils/utils.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cu
)

add_library(cutils ${SOURCE_FILES})
target_include_directories(cutils PRIVATE include/nusol)
target_include_directories(cutils PUBLIC include)
target_compile_options(cutils PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_link_libraries(cutils PUBLIC ${TORCH_LIBRARIES})
set_property(TARGET cutils PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cutils PRIVATE -fPIC)
