set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/container/container.h)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/container.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/entries.cxx
)

add_library(ccontainer STATIC ${SOURCE_FILES})
target_include_directories(ccontainer PRIVATE include/container)
target_include_directories(ccontainer PUBLIC include)
target_link_libraries(ccontainer LINK_PUBLIC cgraph cevent cdataloader)
target_compile_options(ccontainer PRIVATE -fPIC)

