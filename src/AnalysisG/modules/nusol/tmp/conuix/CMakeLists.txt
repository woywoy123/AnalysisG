set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/conics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/matrix.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/conuix.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/matrix.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/nuclx.cxx
)

add_library(cconuix ${SOURCE_FILES})

target_include_directories(cconuix PRIVATE include/conics)
target_include_directories(cconuix PUBLIC include)

target_link_libraries(cconuix PUBLIC cstructs cparticle_template)
target_link_libraries(cconuix PUBLIC cstructs cnusol)

target_compile_options(cconuix PRIVATE -fPIC)
target_compile_options(cconuix PRIVATE -Wall -pedantic -Wextra -Wshadow)
