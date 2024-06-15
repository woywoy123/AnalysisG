set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/generators/graphgenerator.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/graphgenerator.cxx)

add_library(cgraphgenerator STATIC ${SOURCE_FILES})
target_include_directories(cgraphgenerator PRIVATE include/generators)
target_include_directories(cgraphgenerator PUBLIC include)
target_link_libraries(cgraphgenerator LINK_PUBLIC ceventgenerator csampletracer)
target_compile_options(cgraphgenerator PRIVATE -fPIC)

