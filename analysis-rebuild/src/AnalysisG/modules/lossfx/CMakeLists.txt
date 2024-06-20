set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/lossfx.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/fx_enums.h
)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/lossfx.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/switching.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/optimizer_config.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/loss_config.cxx
)

add_library(clossfx STATIC ${SOURCE_FILES})
target_include_directories(clossfx PRIVATE include/templates)
target_include_directories(clossfx PUBLIC include)
target_link_libraries(clossfx LINK_PUBLIC ${TORCH_LIBRARIES} cstructs ctools)
target_compile_options(clossfx PRIVATE -fPIC)
