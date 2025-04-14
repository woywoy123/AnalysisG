# --------------- DEFINE THE EVENT ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/<event-module>/<event-name>.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/<event-name>.cxx)

add_library(c<event-name> STATIC ${SOURCE_FILES})
target_include_directories(c<event-name> PRIVATE include/<event-module>)
target_include_directories(c<event-name> PUBLIC include)

target_link_libraries(c<event-name>  PUBLIC c<particle-module> cevent)
target_compile_options(c<event-name> PRIVATE -fPIC)

cmake_language(CALL cybuild "events/<event-name>" "events/<event-name>" <event-name> "")

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )


