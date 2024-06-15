set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/model_template.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/enum_functions.h
)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/model_template.cxx)

add_library(cmodel STATIC ${SOURCE_FILES})
target_include_directories(cmodel PRIVATE include/templates)
target_include_directories(cmodel PUBLIC include)
target_link_libraries(cmodel LINK_PUBLIC cgraph cnotification)
target_compile_options(cmodel PRIVATE -fPIC)
