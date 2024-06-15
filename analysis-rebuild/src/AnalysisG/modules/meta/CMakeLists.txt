set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/meta/meta.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/meta.cxx)

add_library(cmeta STATIC ${SOURCE_FILES})
target_include_directories(cmeta PRIVATE include/meta)
target_include_directories(cmeta PUBLIC include)
target_link_libraries(cmeta LINK_PUBLIC cstructs RapidJSON ctools)
target_compile_options(cmeta PRIVATE -fPIC)
