set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/selection_template.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/selection_template.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/properties.cxx
)

add_library(cselection STATIC ${SOURCE_FILES})
target_include_directories(cselection PRIVATE include/templates)
target_include_directories(cselection PUBLIC include)
target_link_libraries(cselection LINK_PUBLIC cevent)
target_compile_options(cselection PRIVATE -fPIC)
