set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/settings.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/particles.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/property.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/event.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/meta.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/report.h
)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/structs.cxx)
add_library(cstructs STATIC ${SOURCE_FILES})
target_include_directories(cstructs PRIVATE include/structs)
target_include_directories(cstructs PUBLIC include)
target_compile_options(cstructs PRIVATE -fPIC)
