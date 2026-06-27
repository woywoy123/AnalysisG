set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/reconstruction/pagerank.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/pagerank.cxx)

add_library(creconstruction STATIC ${SOURCE_FILES})
target_include_directories(creconstruction PRIVATE include/reconstruction)
target_include_directories(creconstruction PUBLIC include)
target_link_libraries(creconstruction PUBLIC cparticle_template cnotification)
target_compile_options(creconstruction PRIVATE -fPIC -Wshadow)
