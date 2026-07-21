set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/base.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/main.cxx)

add_library(ag_module_cstructs_base ${SOURCE_FILES})
target_include_directories(ag_module_cstructs_base PRIVATE include/structs)
target_include_directories(ag_module_cstructs_base PUBLIC include)

target_link_libraries(ag_module_cstructs_base PUBLIC ag_module_ctools ag_module_cstructs_enums)
target_compile_options(ag_module_cstructs_base PRIVATE -fPIC -Wshadow)
