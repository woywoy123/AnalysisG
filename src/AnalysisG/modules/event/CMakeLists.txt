set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/event_template.h)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/event_template.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/properties.cxx
)

add_library(cevent STATIC ${SOURCE_FILES})
target_include_directories(cevent PRIVATE include/templates)
target_include_directories(cevent PUBLIC include)
target_link_libraries(cevent LINK_PUBLIC cstructs cparticles ctools cio)
target_compile_options(cevent PRIVATE -fPIC)
