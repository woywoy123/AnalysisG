set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/models/<model-name>.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/<model-name>.cxx)
add_library(c<model-name> STATIC ${SOURCE_FILES})
target_include_directories(c<model-name> PRIVATE include/models)
target_include_directories(c<model-name> PUBLIC include)

target_link_libraries(c<model-name>  PUBLIC cmodel <dependencies>)
target_compile_options(c<model-name> PRIVATE -fPIC)

add_custom_command(OUTPUT <model-name>.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<model-name>.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<model-name>.pyx
VERBATIM COMMAND Python::Interpreter -m cython 
    --cplus ${CMAKE_CURRENT_SOURCE_DIR}/<model-name>.pyx 
    --output-file <model-name>.cpp
)

python_add_library(<model-name>    MODULE <model-name>.cpp WITH_SOABI)
target_link_libraries(<model-name> PUBLIC c<model-name>)

install(TARGETS <model-name> DESTINATION ${SKBUILD_PROJECT_NAME}/model)
install(FILES __init__.pxd DESTINATION . )
install(FILES __init__.py  DESTINATION . )

