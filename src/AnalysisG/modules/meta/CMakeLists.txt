set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/meta/meta.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/meta.cxx)

add_library(cmeta STATIC ${SOURCE_FILES})
target_include_directories(cmeta PRIVATE include/meta)
target_include_directories(cmeta PUBLIC include ${RAPIDJSON_INCLUDE_DIR})
target_link_libraries(cmeta PUBLIC cstructs ctools cnotification ROOT::Core)
target_compile_options(cmeta PRIVATE -fPIC)
target_compile_options(cmeta PRIVATE -Wall -pedantic -Wextra -Wshadow)
