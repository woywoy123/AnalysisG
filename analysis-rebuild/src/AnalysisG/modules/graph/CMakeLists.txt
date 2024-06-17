set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/graph_template.h)
set(SOURCE_FILES 
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/graph_template.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/struct_graph.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/properties.cxx
)

add_library(cgraph STATIC ${SOURCE_FILES})
target_include_directories(cgraph PRIVATE include/templates)
target_include_directories(cgraph PUBLIC include)
target_link_libraries(cgraph LINK_PUBLIC cevent ${TORCH_LIBRARIES})
target_compile_options(cgraph PRIVATE -fPIC)
