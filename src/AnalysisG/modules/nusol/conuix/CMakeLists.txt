set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/solvers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/struct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/matrix.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/memory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/conuic.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/htilde.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/struct.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/matrix.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/memory.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/conuic.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/eigen.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/htilde.cxx
)

add_library(sconuix ${SOURCE_FILES})
target_include_directories(sconuix PRIVATE include/conuix)
target_include_directories(sconuix PUBLIC include)
target_link_libraries(sconuix  PRIVATE cparticle_template cstructs)

target_compile_options(sconuix PRIVATE -fPIC)
target_compile_options(sconuix PRIVATE -Wall -pedantic -Wextra -Wshadow)


set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/conuix.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/conuix.cxx
)

add_library(cconuix ${SOURCE_FILES})
target_include_directories(cconuix PRIVATE include/conuix)
target_include_directories(cconuix PUBLIC include)

target_link_libraries(cconuix  PRIVATE cstructs cnusol sconuix)
target_compile_options(cconuix PRIVATE -fPIC)
target_compile_options(cconuix PRIVATE -Wall -pedantic -Wextra -Wshadow)
