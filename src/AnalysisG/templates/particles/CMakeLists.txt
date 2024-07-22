# --------------- DEFINE THE PARTICLES ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/<particle-module>/<particle-name>.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/<particle-name>.cxx)
add_library(c<particle-name> STATIC ${SOURCE_FILES})
target_link_libraries(c<particle-name> LINK_PUBLIC c<particle-module> cevent)
target_compile_options(c<particle-name> PRIVATE -fPIC)
target_include_directories(c<particle-name> PUBLIC include)
target_include_directories(c<particle-name> PRIVATE include/<particle-module>)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/<particle-name>.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<particle-name>.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/<particle-name>.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/<particle-name>.pyx 
    --output-file ${CMAKE_CURRENT_SOURCE_DIR}/<particle-name>.cpp
)

python_add_library(<particle-name> MODULE ${CMAKE_CURRENT_SOURCE_DIR}/<particle-name>.cpp WITH_SOABI)
target_link_libraries(<particle-name> LINK_PUBLIC c<event-name>)
install(FILES __init__.pxd          DESTINATION ${SKBUILD_PROJECT_NAME}/events/<particle-name>)
install(TARGETS <particle-name>     DESTINATION ${SKBUILD_PROJECT_NAME}/events/<particle-name>)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/<particle-name>.cpp)

