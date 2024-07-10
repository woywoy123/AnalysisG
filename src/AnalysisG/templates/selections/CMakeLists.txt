set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/<selection-name>.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/<selection-name>.cxx)

add_library(c<selection-package-name> STATIC ${SOURCE_FILES})
target_include_directories(c<selection-package-name> PUBLIC ../<selection-name>)
target_link_libraries(c<selection-package-name> LINK_PUBLIC cselection <dependencies>)
target_compile_options(c<selection-package-name> PRIVATE -fPIC)

add_custom_command(
OUTPUT <selection-name>.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<selection-name>.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<selection-name>.pyx
VERBATIM
COMMAND 
    Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/<selection-name>.pyx 
    --output-file <selection-name>.cpp
)

python_add_library(<selection-name> MODULE <selection-name>.cpp WITH_SOABI)
target_link_libraries(<selection-name> LINK_PUBLIC c<selection-package-name>)
install(FILES        __init__.pxd  DESTINATION ${SKBUILD_PROJECT_NAME}/selections/<output-path>)
install(TARGETS   <selection-name> DESTINATION ${SKBUILD_PROJECT_NAME}/selections/<output-path>)


