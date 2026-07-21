set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/tracing.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/main.cxx)

add_library(ag_module_cstructs_tracing ${SOURCE_FILES})
target_include_directories(ag_module_cstructs_tracing PRIVATE include/structs)
target_include_directories(ag_module_cstructs_tracing PUBLIC include)

target_link_libraries(ag_module_cstructs_tracing PUBLIC ag_module_ctools ag_module_cstructs_multithreaded)
target_compile_options(ag_module_cstructs_tracing PRIVATE -fPIC -Wshadow)
