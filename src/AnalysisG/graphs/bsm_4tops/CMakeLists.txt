set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/graph_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/node_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/edge_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/graphs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/graphs.cxx
)

add_library(cgraph_bsm_4tops                STATIC ${SOURCE_FILES})
target_include_directories(cgraph_bsm_4tops PRIVATE include/bsm_4tops)
target_include_directories(cgraph_bsm_4tops PUBLIC include)

target_link_libraries(cgraph_bsm_4tops  PUBLIC cevent_bsm_4tops cgraph_template)
target_compile_options(cgraph_bsm_4tops PRIVATE -fPIC)

cmake_language(CALL cybuild "graphs/bsm_4tops" "graphs/bsm_4tops" graph_bsm_4tops "")

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

