set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/multisol/multisol.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/multisol/matrix.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/multisol.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/matrix.cxx
)

add_library(cmultisol ${SOURCE_FILES})
target_include_directories(cmultisol PRIVATE include/multisol)
target_include_directories(cmultisol PUBLIC include)
target_link_libraries(cmultisol  PUBLIC cstructs cparticle_template)
target_compile_options(cmultisol PRIVATE -fPIC)
target_compile_options(cmultisol PRIVATE -Wall -pedantic -Wextra -Wshadow)
