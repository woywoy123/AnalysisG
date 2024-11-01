set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/plotting/plotting.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/plotting.cxx)

add_library(cplotting STATIC ${SOURCE_FILES})
target_include_directories(cplotting PRIVATE include/plotting)
target_include_directories(cplotting PUBLIC include)
target_link_libraries(cplotting LINK_PUBLIC ctools cstructs cnotification ${ROOT_LIBRARIES})
target_compile_options(cplotting PRIVATE -fPIC)
