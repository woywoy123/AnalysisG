
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/<selection-name>.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/<selection-name>.cxx)

add_library(c<selection-name> STATIC ${SOURCE_FILES})
target_include_directories(c<selection-name> PUBLIC ../<selection-name>)
target_link_libraries(c<selection-name>  PUBLIC cselection_template <dependencies>)
target_compile_options(c<selection-name> PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/<name>/<selection-name>" "selections/<selection-name>" <selection-name> "")
