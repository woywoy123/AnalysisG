add_subdirectory(bsm_4tops)

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

python_add_library(graph_bsm_4tops MODULE ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.cpp WITH_SOABI)
target_link_libraries(graph_bsm_4tops LINK_PUBLIC cgraph_bsm_4tops)
set_property(TARGET graph_bsm_4tops APPEND PROPERTY ADDITIONAL_CLEAN_FILES bsm_4tops.cpp)

install(FILES __init__.pxd       DESTINATION ${SKBUILD_PROJECT_NAME}/graphs)
install(TARGETS graph_bsm_4tops  DESTINATION ${SKBUILD_PROJECT_NAME}/graphs)

