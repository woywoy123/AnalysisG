set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/reconstruction/nusol.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/nusol.cxx)

add_library(rnusol ${SOURCE_FILES})
target_include_directories(rnusol PRIVATE include/reconstruction)
target_include_directories(rnusol PUBLIC include)

target_link_libraries(rnusol  PUBLIC cstructs ctools cnotification cparticle_template rellipse rconuix rconuic ccommon)

target_compile_options(rnusol PRIVATE -fPIC)
target_compile_options(rnusol PRIVATE -Wall -pedantic -Wextra -Wshadow)
