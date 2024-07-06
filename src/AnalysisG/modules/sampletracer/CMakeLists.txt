set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/generators/sampletracer.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/sampletracer.cxx)

add_library(csampletracer STATIC ${SOURCE_FILES})
target_include_directories(csampletracer PRIVATE include/generators)
target_include_directories(csampletracer PUBLIC include)
target_link_libraries(csampletracer LINK_PUBLIC ctools ccontainer)
target_compile_options(csampletracer PRIVATE -fPIC)
