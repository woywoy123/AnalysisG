set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conics/conics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conics/matrix.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/conics.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/matrix.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/nuclx.cxx
)

add_library(cconics ${SOURCE_FILES})

target_include_directories(cconics PRIVATE include/conics)
target_include_directories(cconics PUBLIC include)

target_link_libraries(cconics PUBLIC cstructs cparticle_template)
target_link_libraries(cconics PUBLIC cstructs cnusol)

target_compile_options(cconics PRIVATE -fPIC)
target_compile_options(cconics PRIVATE -Wall -pedantic -Wextra -Wshadow)
