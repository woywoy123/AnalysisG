set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/model_template.h)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/model_template.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/model_configuration.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/model_lossfx.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/model_checks.cxx
)

add_library(cmodel_template STATIC ${SOURCE_FILES})
target_include_directories(cmodel_template PRIVATE include/templates)
target_include_directories(cmodel_template PUBLIC include)
target_link_libraries(cmodel_template PUBLIC cgraph_template cnotification clossfx)

target_compile_options(cmodel_template PRIVATE -fPIC)
target_compile_options(cmodel_template PRIVATE -Wall -pedantic -Wextra -Wshadow)
if(CMAKE_ANALYSISG_CUDA)
target_compile_definitions(cmodel_template PRIVATE PYC_CUDA ON)
endif()


