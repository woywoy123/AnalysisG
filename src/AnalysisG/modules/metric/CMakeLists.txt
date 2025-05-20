set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/metric_template.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/metric_template.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/properties.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/link.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/base.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/metric.cxx
)

add_library(cmetric_template STATIC ${SOURCE_FILES})
target_include_directories(cmetric_template PRIVATE include/templates)
target_include_directories(cmetric_template PUBLIC include)
target_link_libraries(cmetric_template PUBLIC ctypecasting cmodel_template cio cplotting cparticle_template)
target_compile_options(cmetric_template PRIVATE -fPIC)
target_compile_options(cmetric_template PRIVATE -Wall -pedantic -Wextra -Wshadow)
