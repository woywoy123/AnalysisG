# --------------- DEFINE THE EVENT ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/<event-module>/<event-name>.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/<event-name>.cxx)

add_library(c<event-name> STATIC ${SOURCE_FILES})
target_include_directories(c<event-name> PRIVATE include/<event-module>)
target_include_directories(c<event-name> PUBLIC include)

target_link_libraries(c<event-name>  PUBLIC c<particle-module> cevent)
target_compile_options(c<event-name> PRIVATE -fPIC)

add_custom_command(OUTPUT <event-name>.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<event-name>.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<event-name>.pyx
VERBATIM COMMAND Python::Interpreter -m cython 
    --cplus ${CMAKE_CURRENT_SOURCE_DIR}/<event-name>.pyx 
    --output-file <event-name>.cpp
)

python_add_library(<event-name>    MODULE <event-name>.cpp WITH_SOABI)
target_link_libraries(<event-name> PUBLIC c<event-name>)
install(TARGETS <event-name> DESTINATION ${SKBUILD_PROJECT_NAME}/events/<event-module>)

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )


