# --------------- DEFINE THE PARTICLES ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/<particle-module>/<particle-name>.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/<particle-name>.cxx)

add_library(c<particle-name> STATIC ${SOURCE_FILES})
target_include_directories(c<particle-name> PRIVATE include/<particle-module>)
target_include_directories(c<particle-name> PUBLIC include)

target_link_libraries(c<particle-name>  PUBLIC cparticles)
target_compile_options(c<particle-name> PRIVATE -fPIC)

add_custom_command(OUTPUT <particle-name>.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<particle-name>.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<particle-name>.pyx
VERBATIM COMMAND Python::Interpreter -m cython 
    --cplus ${CMAKE_CURRENT_SOURCE_DIR}/<particle-name>.pyx 
    --output-file <particle-name>.cpp
)

python_add_library(<particle-name>    MODULE <particle-name>.cpp WITH_SOABI)
target_link_libraries(<particle-name> PUBLIC c<particle-name>)

install(TARGETS <particle-name> DESTINATION ${SKBUILD_PROJECT_NAME}/events/<particle-name>)
install(FILES __init__.pxd DESTINATION . )
install(FILES __init__.py  DESTINATION . )


