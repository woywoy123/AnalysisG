set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/transform/polar-cuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/polar.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/polar.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/torch.cu
)

add_library(polar-cuda STATIC ${SOURCE_FILES})
target_include_directories(polar-cuda PRIVATE include/transform)
target_include_directories(polar-cuda PUBLIC include)
target_link_libraries(polar-cuda PUBLIC ${TORCH_LIBRARIES})
target_compile_options(polar-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_compile_options(polar-cuda PRIVATE -fPIC)
