set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/typecasting/detensor.h
)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/main.cxx
)

add_library(ag_module_ccasting_detensor STATIC ${SOURCE_FILES})
target_include_directories(ag_module_ccasting_detensor PRIVATE include/typecasting)
target_include_directories(ag_module_ccasting_detensor PUBLIC include)

target_link_libraries(ag_module_ccasting_detensor PUBLIC ag_module_ctools ag_module_cstructs_enums)
target_link_libraries(ag_module_ccasting_detensor PUBLIC ${TORCH_LIBRARIES})

target_compile_options(ag_module_ccasting_detensor PRIVATE -fPIC)
target_compile_options(ag_module_ccasting_detensor PRIVATE -Wall -pedantic -Wextra -Wshadow)

