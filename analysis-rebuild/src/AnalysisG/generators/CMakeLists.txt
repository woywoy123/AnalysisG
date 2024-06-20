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

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/coptimizer.cpp
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/coptimizer.cpp
)

python_add_library(analysis    MODULE ${CMAKE_CURRENT_SOURCE_DIR}/canalysis.cpp WITH_SOABI)
target_link_libraries(analysis LINK_PUBLIC canalysis)

python_add_library(optimizer    MODULE ${CMAKE_CURRENT_SOURCE_DIR}/coptimizer.cpp WITH_SOABI)
target_link_libraries(optimizer LINK_PUBLIC coptimizer)

set_property(TARGET analysis  APPEND PROPERTY ADDITIONAL_CLEAN_FILES canalysis.cpp)
set_property(TARGET optimizer APPEND PROPERTY ADDITIONAL_CLEAN_FILES coptimizer.cpp)

install(FILES __init__.pxd DESTINATION ${SKBUILD_PROJECT_NAME}/generators)
install(TARGETS analysis   DESTINATION ${SKBUILD_PROJECT_NAME}/generators)
install(TARGETS optimizer  DESTINATION ${SKBUILD_PROJECT_NAME}/generators)

