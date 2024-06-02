set(HEADER_FILES include/tools/tools.h)
set(SOURCE_FILES 
        tools.cxx
        io.cxx
        strings.cxx
        conversion.cxx
)
add_library(tools STATIC ${SOURCE_FILES})
target_include_directories(tools PRIVATE include/tools)
target_include_directories(tools PUBLIC include)
target_link_libraries(tools LINK_PRIVATE uuid)
target_compile_options(tools PRIVATE -fPIC)
