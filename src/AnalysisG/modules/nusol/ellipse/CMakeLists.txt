set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ellipse/ellipse.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ellipse/solvers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ellipse/nusol.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ellipse/mtx.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/ellipse.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/solvers.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/nusol.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/mtx.cxx
)

add_library(cellipse ${SOURCE_FILES})
target_include_directories(cellipse PRIVATE include/ellipse)
target_include_directories(cellipse PUBLIC include)

target_link_libraries(cellipse  PUBLIC cstructs cparticle_template)
target_link_libraries(cellipse  PRIVATE cnusol sconuix)

target_compile_options(cellipse PRIVATE -fPIC)
target_compile_options(cellipse PRIVATE -Wall -pedantic -Wextra -Wshadow)
