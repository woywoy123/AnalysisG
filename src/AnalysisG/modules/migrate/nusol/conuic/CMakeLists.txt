set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/variables.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/constants.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/branches.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/angular.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/atomics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/pencil.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/factor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/base.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/variables.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/constants.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/branches.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/angular.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/atomics.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/pencil.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/factor.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/base.cxx
)

add_library(srconuic ${SOURCE_FILES})
target_include_directories(srconuic PRIVATE include/conuic)
target_include_directories(srconuic PUBLIC include)
target_link_libraries(srconuic  PRIVATE cparticle_template cstructs ccommon)

target_compile_options(srconuic PRIVATE -fPIC)
target_compile_options(srconuic PRIVATE -Wall -pedantic -Wextra -Wshadow)


set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/conuic.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/conuic.cxx
)

add_library(rconuic ${SOURCE_FILES})
target_include_directories(rconuic PRIVATE include/conuic)
target_include_directories(rconuic PUBLIC include)

target_link_libraries(rconuic  PRIVATE cstructs rnusol srconuic ccommon)
target_compile_options(rconuic PRIVATE -fPIC)
target_compile_options(rconuic PRIVATE -Wall -pedantic -Wextra -Wshadow)
