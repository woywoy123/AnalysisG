set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/typecasting/elements.h
)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/main.cxx
)

add_library(ag_module_ccasting_shared STATIC ${SOURCE_FILES})
target_include_directories(ag_module_ccasting_shared PRIVATE include/typecasting)
target_include_directories(ag_module_ccasting_shared PUBLIC include)

target_link_libraries(ag_module_ccasting_shared PUBLIC ag_module_ctools ag_module_cstructs_enums)
target_link_libraries(ag_module_ccasting_shared PUBLIC ${TORCH_LIBRARIES} ${ROOT_LIBRARIES})

target_compile_options(ag_module_ccasting_shared PRIVATE -fPIC)
#if(CMAKE_ANALYSISG_CUDA)
#target_compile_definitions(ctypecasting PRIVATE PYC_CUDA ON)
#endif()
target_compile_options(ag_module_ccasting_shared PRIVATE -Wall -pedantic -Wextra -Wshadow)

