set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/generators/eventgenerator.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/eventgenerator.cxx)

add_library(ceventgenerator STATIC ${SOURCE_FILES})
target_include_directories(ceventgenerator PRIVATE include/generators)
target_include_directories(ceventgenerator PUBLIC include)
target_link_libraries(ceventgenerator LINK_PUBLIC ctools cio cnotification cevent)
target_compile_options(ceventgenerator PRIVATE -fPIC)

