set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/structs.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/element.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/optimizer.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/misc.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/base.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/variable.cxx
)

add_library(cstructs ${SOURCE_FILES})
target_include_directories(cstructs PRIVATE include/structs)
target_include_directories(cstructs PUBLIC include ${ROOT_INCLUDE_DIRS})
target_link_libraries(cstructs PUBLIC ${ROOT_LIBRARIES} ctypecasting)
target_compile_options(cstructs PRIVATE -fPIC)
