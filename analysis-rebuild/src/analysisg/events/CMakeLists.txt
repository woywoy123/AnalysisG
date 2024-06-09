add_subdirectory(bsm_4tops)


add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.cpp
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.cpp
)

python_add_library(bsm_4tops MODULE ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.cpp WITH_SOABI)
target_link_libraries(bsm_4tops LINK_PUBLIC cbsm_4tops)

install(FILES __init__.pxd DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/events)
install(TARGETS bsm_4tops  DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/events)

