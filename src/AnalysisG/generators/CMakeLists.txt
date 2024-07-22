add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/canalysis.cpp
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/analysis.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/analysis.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        ${CMAKE_CURRENT_SOURCE_DIR}/analysis.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/canalysis.cpp
)

python_add_library(analysis    MODULE ${CMAKE_CURRENT_SOURCE_DIR}/canalysis.cpp WITH_SOABI)
target_link_libraries(analysis LINK_PUBLIC canalysis)
install(FILES __init__.pxd DESTINATION ${SKBUILD_PROJECT_NAME}/generators)
install(FILES __init__.py  DESTINATION ${SKBUILD_PROJECT_NAME}/generators)
install(TARGETS analysis   DESTINATION ${SKBUILD_PROJECT_NAME}/generators)

