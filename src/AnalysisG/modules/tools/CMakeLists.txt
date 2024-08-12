set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/tools.h)
set(SOURCE_FILES
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/tools.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/strings.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/io.cxx
)

add_library(ctools STATIC ${SOURCE_FILES})
target_include_directories(ctools PRIVATE include/tools)
target_include_directories(ctools PUBLIC include)
target_link_libraries(ctools  PUBLIC stdc++fs)
target_compile_options(ctools PRIVATE -fPIC)

