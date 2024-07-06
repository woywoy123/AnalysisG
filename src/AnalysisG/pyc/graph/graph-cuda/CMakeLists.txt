set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/graph-cuda.h
)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/graph.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/graph-kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/graph.cxx
)

add_library(graph-cuda STATIC ${SOURCE_FILES})
target_include_directories(graph-cuda PRIVATE include/graph)
target_include_directories(graph-cuda PUBLIC include)
target_link_libraries(graph-cuda LINK_PUBLIC physics-cuda polar-cuda cartesian-cuda)
target_compile_options(graph-cuda PRIVATE -fPIC)
