# --------------- DEFINE THE EVENT ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/<event-module>/<event-name>.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/<event-name>.cxx)
add_library(c<event-name> STATIC ${SOURCE_FILES})
target_link_libraries(c<event-name> LINK_PUBLIC c<particle-module> cevent)
target_compile_options(c<event-name> PRIVATE -fPIC)
target_include_directories(c<event-name> PUBLIC include)
target_include_directories(c<event-name> PRIVATE include/<event-module>)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/<event-name>.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<event-name>.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<event-name>.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/<event-name>.pyx 
    --output-file ${CMAKE_CURRENT_SOURCE_DIR}/<event-name>.cpp
)

python_add_library(<event-name> MODULE ${CMAKE_CURRENT_SOURCE_DIR}/<event-name>.cpp WITH_SOABI)
target_link_libraries(<event-name> LINK_PUBLIC c<event-name>)
install(FILES __init__.pxd          DESTINATION ${SKBUILD_PROJECT_NAME}/events/<event-module>)
install(TARGETS <event-name>     DESTINATION ${SKBUILD_PROJECT_NAME}/events/<event-module>)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/<event-name>.cpp)

