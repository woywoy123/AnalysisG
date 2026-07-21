set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/plotting/roc.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/roc.cxx)

add_library(croc STATIC ${SOURCE_FILES})
target_include_directories(croc PRIVATE include/plotting)
target_include_directories(croc PUBLIC include)
target_link_libraries(croc PUBLIC cplotting ${ROOT_LIBRARIES})
target_compile_options(croc PRIVATE -fPIC -Wshadow)
