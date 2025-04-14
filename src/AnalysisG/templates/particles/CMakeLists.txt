# --------------- DEFINE THE PARTICLES ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/<particle-module>/<particle-name>.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/<particle-name>.cxx)

add_library(c<particle-name> STATIC ${SOURCE_FILES})
target_include_directories(c<particle-name> PRIVATE include/<particle-module>)
target_include_directories(c<particle-name> PUBLIC include)

target_link_libraries(c<particle-name>  PUBLIC cparticle_template)
target_compile_options(c<particle-name> PRIVATE -fPIC)

cmake_language(CALL cybuild "events/<event-name>" "events/<event-name>" <particle-name> "")

install(TARGETS <particle-name> DESTINATION ${SKBUILD_PROJECT_NAME}/events/<particle-name>)
install(FILES __init__.pxd DESTINATION . )
install(FILES __init__.py  DESTINATION . )

