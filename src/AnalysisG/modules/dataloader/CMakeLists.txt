set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/generators/dataloader.h)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/dataloader.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/cache.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/dataset.cxx
)

add_library(cdataloader STATIC ${SOURCE_FILES})
target_include_directories(cdataloader PRIVATE include/generators)
target_include_directories(cdataloader PUBLIC include)
target_link_libraries(cdataloader LINK_PUBLIC cgraph_template cnotification ctools cstructs cio)
target_compile_options(cdataloader PRIVATE -fPIC)

