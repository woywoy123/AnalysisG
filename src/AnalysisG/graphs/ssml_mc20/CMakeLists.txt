set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ssml_mc20/graph_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ssml_mc20/node_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ssml_mc20/edge_features.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ssml_mc20/graphs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/graphs.cxx
)

add_library(cgraph_ssml_mc20                STATIC ${SOURCE_FILES})
target_include_directories(cgraph_ssml_mc20 PUBLIC include)
target_include_directories(cgraph_ssml_mc20 PRIVATE include/ssml_mc20)
target_link_libraries(cgraph_ssml_mc20      LINK_PUBLIC cevent_ssml_mc20 cgraph)
target_compile_options(cgraph_ssml_mc20     PRIVATE -fPIC)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/ssml_mc20.cpp
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_ssml_mc20.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_ssml_mc20.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_ssml_mc20.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/ssml_mc20.cpp
)

python_add_library(graph_ssml_mc20    MODULE ${CMAKE_CURRENT_SOURCE_DIR}/ssml_mc20.cpp WITH_SOABI)
target_link_libraries(graph_ssml_mc20 LINK_PUBLIC cgraph_ssml_mc20)
install(TARGETS graph_ssml_mc20       DESTINATION ${SKBUILD_PROJECT_NAME}/graphs/ssml_mc20)

file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/ssml_mc20.cpp)
file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

