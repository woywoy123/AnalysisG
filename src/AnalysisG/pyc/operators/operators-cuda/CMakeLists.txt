set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/operators/operators-cuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/operators/operators.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/operators.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/torch.cu
)

add_library(operators-cuda STATIC ${SOURCE_FILES})
target_include_directories(operators-cuda PRIVATE include/operators)
target_include_directories(operators-cuda PUBLIC include)
target_link_libraries(operators-cuda PUBLIC ${TORCH_LIBRARIES})
target_compile_options(operators-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_compile_options(operators-cuda PRIVATE -fPIC)

