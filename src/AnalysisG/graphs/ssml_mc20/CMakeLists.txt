set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ssml_mc20/graph_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ssml_mc20/node_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ssml_mc20/edge_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ssml_mc20/graphs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/graphs.cxx
)

add_library(cgraph_ssml_mc20                STATIC ${SOURCE_FILES})
target_include_directories(cgraph_ssml_mc20 PRIVATE include/ssml_mc20)
target_include_directories(cgraph_ssml_mc20 PUBLIC include)

target_link_libraries(cgraph_ssml_mc20  PUBLIC cevent_ssml_mc20 cgraph_template)
target_compile_options(cgraph_ssml_mc20 PRIVATE -fPIC)

cmake_language(CALL cybuild "graphs/ssml_mc20" "graphs/ssml_mc20" graph_ssml_mc20 "")
file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

