set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/container/container.h)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/container.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/entries.cxx
)

add_library(ccontainer STATIC ${SOURCE_FILES})
target_include_directories(ccontainer PRIVATE include/container)
target_include_directories(ccontainer PUBLIC include)
target_link_libraries(ccontainer PUBLIC cgraph_template cevent_template cdataloader cselection_template)
target_compile_options(ccontainer PRIVATE -fPIC)
target_compile_options(ccontainer PRIVATE -Wall -pedantic -Wextra)

