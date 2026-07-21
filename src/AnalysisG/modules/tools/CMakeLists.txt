set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/tools.h
)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/io.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/tools.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/strings.cxx
)

add_library(ag_module_ctools STATIC ${SOURCE_FILES})
target_include_directories(ag_module_ctools PRIVATE include/tools)
target_include_directories(ag_module_ctools PUBLIC include)

target_compile_options(ag_module_ctools PRIVATE -fPIC)
target_compile_options(ag_module_ctools PRIVATE -Wall -pedantic -Wextra -Wshadow)

