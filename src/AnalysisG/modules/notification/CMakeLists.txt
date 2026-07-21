set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/modules/notification.h
)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/notification.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/formats.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/eol.cxx
)

add_library(ag_module_cnotification STATIC ${SOURCE_FILES})
target_include_directories(ag_module_cnotification PRIVATE include/modules)
target_include_directories(ag_module_cnotification PUBLIC include)

target_link_libraries(ag_module_cnotification 
    PUBLIC 
        ag_module_ctools 
        ag_module_cstructs_tracing 
        ag_module_cstructs_mulithreaded
)

target_compile_options(ag_module_cnotification PRIVATE -fPIC)
target_compile_options(ag_module_cnotification PRIVATE -Wall -pedantic -Wextra -Wshadow)
