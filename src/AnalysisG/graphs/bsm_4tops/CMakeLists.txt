set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/graph_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/node_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/edge_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/graphs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/graphs.cxx
)

add_library(cgraph_bsm_4tops                STATIC ${SOURCE_FILES})
target_include_directories(cgraph_bsm_4tops PUBLIC include)
target_include_directories(cgraph_bsm_4tops PRIVATE include/bsm_4tops)
target_link_libraries(cgraph_bsm_4tops      LINK_PUBLIC cevent_bsm_4tops cgraph)
target_compile_options(cgraph_bsm_4tops     PRIVATE -fPIC)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.cpp
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_bsm_4tops.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_bsm_4tops.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_bsm_4tops.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.cpp
)

python_add_library(graph_bsm_4tops    MODULE ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.cpp WITH_SOABI)
target_link_libraries(graph_bsm_4tops LINK_PUBLIC cgraph_bsm_4tops)
install(TARGETS graph_bsm_4tops       DESTINATION ${SKBUILD_PROJECT_NAME}/graphs/bsm_4tops)

file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.cpp)
file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )




