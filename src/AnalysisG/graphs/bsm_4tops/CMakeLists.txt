set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/node_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/edge_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/graphs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/graphs.cxx
)

add_library(cgraph_bsm_4tops STATIC ${SOURCE_FILES})
target_link_libraries(cgraph_bsm_4tops LINK_PUBLIC cevent_bsm_4tops cgraph)
target_compile_options(cgraph_bsm_4tops PRIVATE -fPIC)
