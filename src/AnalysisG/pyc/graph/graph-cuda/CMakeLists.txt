set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/graph-cuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/graph.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/graph-kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/graph.cxx
)

add_library(graph-cuda STATIC ${SOURCE_FILES})
target_include_directories(graph-cuda PRIVATE include/graph)
target_include_directories(graph-cuda PUBLIC include)
target_link_libraries(graph-cuda PUBLIC physics-cuda polar-cuda cartesian-cuda)
set_property(TARGET graph-cuda PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(graph-cuda PRIVATE -fPIC)
