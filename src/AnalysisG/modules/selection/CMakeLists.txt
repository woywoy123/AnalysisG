set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/selection_template.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/selection_template.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/properties.cxx
)

add_library(cselection_template STATIC ${SOURCE_FILES})
target_include_directories(cselection_template PRIVATE include/templates)
target_include_directories(cselection_template PUBLIC include)
target_link_libraries(cselection_template PUBLIC cevent_template ctypecasting cnotification)
target_compile_options(cselection_template PRIVATE -fPIC)
