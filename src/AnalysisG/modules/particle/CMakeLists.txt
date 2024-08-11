set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/particle_template.h)
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/particle_template.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/interfaces.cxx

    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/physics.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/polar.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/cartesian.cxx
)

add_library(cparticle_template STATIC ${SOURCE_FILES})
target_include_directories(cparticle_template PRIVATE include/templates)
target_include_directories(cparticle_template PUBLIC include)
target_link_libraries(cparticle_template  PUBLIC cstructs ctools)
target_compile_options(cparticle_template PRIVATE -fPIC)
