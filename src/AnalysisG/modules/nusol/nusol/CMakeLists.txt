set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/reconstruction/nusol.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/nusol.cxx)

add_library(cnusol ${SOURCE_FILES})
target_include_directories(cnusol PRIVATE include/reconstruction)
target_include_directories(cnusol PUBLIC include)

target_link_libraries(cnusol  PUBLIC cstructs ctools cnotification cparticle_template cellipse cconuix)

target_compile_options(cnusol PRIVATE -fPIC)
target_compile_options(cnusol PRIVATE -Wall -pedantic -Wextra -Wshadow)
