set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/models/<model-name>.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/<model-name>.cxx)

add_library(c<model-name> STATIC ${SOURCE_FILES})
target_include_directories(c<model-name> PRIVATE include/models)
target_include_directories(c<model-name> PUBLIC include)

target_link_libraries(c<model-name>  PUBLIC cmodel <dependencies>)
target_compile_options(c<model-name> PRIVATE -fPIC)

cmake_language(CALL cybuild "models/<model-name>" "models" <model-name> "")

