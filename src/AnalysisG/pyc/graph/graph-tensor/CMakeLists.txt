set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/graph.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/graph.cxx)

add_library(graph-tensor STATIC ${SOURCE_FILES})
target_include_directories(graph-tensor PRIVATE include/graph)
target_include_directories(graph-tensor PUBLIC include)
target_link_libraries(graph-tensor LINK_PUBLIC physics-tensors polar-tensors cartesian-tensors)
