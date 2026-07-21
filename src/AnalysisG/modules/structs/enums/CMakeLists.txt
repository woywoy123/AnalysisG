set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/enums.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/main.cxx)

add_library(ag_module_cstructs_enums ${SOURCE_FILES})
target_include_directories(ag_module_cstructs_enums PRIVATE include/structs)
target_include_directories(ag_module_cstructs_enums PUBLIC include)

target_link_libraries(ag_module_cstructs_enums PUBLIC ag_module_ctools)
target_compile_options(ag_module_cstructs_enums PRIVATE -fPIC -Wshadow)
