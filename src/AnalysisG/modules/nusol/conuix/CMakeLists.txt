set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/struct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/conuic.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/htilde.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/struct.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/conuic.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/eigen.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/htilde.cxx
)

add_library(srconuix ${SOURCE_FILES})
target_include_directories(srconuix PRIVATE include/conuix)
target_include_directories(srconuix PUBLIC include)
target_link_libraries(srconuix  PRIVATE cparticle_template cstructs ccommon)

target_compile_options(srconuix PRIVATE -fPIC)
target_compile_options(srconuix PRIVATE -Wall -pedantic -Wextra -Wshadow)


set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuix/conuix.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/conuix.cxx
)

add_library(rconuix ${SOURCE_FILES})
target_include_directories(rconuix PRIVATE include/conuix)
target_include_directories(rconuix PUBLIC include)

target_link_libraries(rconuix  PRIVATE cstructs rnusol srconuix ccommon)
target_compile_options(rconuix PRIVATE -fPIC)
target_compile_options(rconuix PRIVATE -Wall -pedantic -Wextra -Wshadow)
