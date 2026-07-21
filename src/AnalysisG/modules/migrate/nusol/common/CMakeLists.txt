set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common/solvers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common/matrix.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common/memory.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/matrix.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/memory.cxx
)

add_library(ccommon ${SOURCE_FILES})
target_include_directories(ccommon PRIVATE include/common)
target_include_directories(ccommon PUBLIC include)

target_compile_options(ccommon PRIVATE -fPIC)
target_compile_options(ccommon PRIVATE -Wall -pedantic -Wextra -Wshadow)
