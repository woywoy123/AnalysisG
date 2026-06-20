set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/models/utils.h)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/activations.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/generators.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/functional.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/utils.cxx
)

add_library(cUrecyclx STATIC ${SOURCE_FILES})
target_include_directories(cUrecyclx PRIVATE include/models)
target_include_directories(cUrecyclx PUBLIC include)
target_link_libraries(cUrecyclx PUBLIC cmodel_template cpyc)
target_compile_options(cUrecyclx PRIVATE -fPIC)


set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/models/recyclx.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/recyclx.cxx)

add_library(crecyclx STATIC ${SOURCE_FILES})
target_include_directories(crecyclx PRIVATE include/models)
target_include_directories(crecyclx PUBLIC include)
target_link_libraries(crecyclx PUBLIC cUrecyclx)
target_compile_options(crecyclx PRIVATE -fPIC)
cmake_language(CALL cybuild "models/RecyclX" "models" recyclx "")
