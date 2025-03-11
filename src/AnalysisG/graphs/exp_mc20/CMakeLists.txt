set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/exp_mc20/graph_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/exp_mc20/node_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/exp_mc20/edge_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/exp_mc20/graphs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/graphs.cxx
)

add_library(cgraph_exp_mc20                STATIC ${SOURCE_FILES})
target_include_directories(cgraph_exp_mc20 PRIVATE include/exp_mc20)
target_include_directories(cgraph_exp_mc20 PUBLIC include)

target_link_libraries(cgraph_exp_mc20  PUBLIC cevent_exp_mc20 cgraph_template)
target_compile_options(cgraph_exp_mc20 PRIVATE -fPIC)

cmake_language(CALL cybuild "graphs/exp_mc20" "graphs/exp_mc20" graph_exp_mc20 "")
file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

