set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/graph_template.h)
set(SOURCE_FILES 
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/graph_template.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/struct_graph.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/properties.cxx
)

add_library(cgraph_template STATIC ${SOURCE_FILES})
target_include_directories(cgraph_template PRIVATE include/templates)
target_include_directories(cgraph_template PUBLIC include)
target_link_libraries(cgraph_template LINK_PUBLIC cevent_template ctypecasting nusol-cuda ${TORCH_LIBRARIES})
target_compile_options(cgraph_template PRIVATE -fPIC)
