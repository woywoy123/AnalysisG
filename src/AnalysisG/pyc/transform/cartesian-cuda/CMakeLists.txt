set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/transform/cartesian-cuda.h)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cartesian.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cartesian.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/torch.cu
)

add_library(cartesian-cuda STATIC ${SOURCE_FILES})
target_include_directories(cartesian-cuda PRIVATE include/transform)
target_include_directories(cartesian-cuda PUBLIC include)
target_link_libraries(cartesian-cuda LINK_PUBLIC ${TORCH_LIBRARIES})
target_compile_options(cartesian-cuda PRIVATE -fPIC)
