set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/typecasting/
)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/
)

add_library(ag_module_casting_croot STATIC ${SOURCE_FILES})
target_include_directories(ag_module_casting_croot PRIVATE include/modules)
target_include_directories(ag_module_casting_croot PUBLIC include)

target_link_libraries(ag_module_casting_croot PUBLIC ag_module_ctools)

target_compile_options(ag_module_casting_croot PRIVATE -fPIC)
target_compile_options(ag_module_casting_croot PRIVATE -Wall -pedantic -Wextra -Wshadow)
