set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/structs.cxx)

add_library(cstructs STATIC ${SOURCE_FILES})
target_include_directories(cstructs PRIVATE include/structs)
target_include_directories(cstructs PUBLIC include ${ROOT_INCLUDE_DIRS})
target_link_libraries(cstructs PUBLIC ${ROOT_LIBRARIES})
target_compile_options(cstructs PRIVATE -fPIC)
