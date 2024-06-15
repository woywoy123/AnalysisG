set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/particle_template.h)
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/particle_template.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/interfaces.cxx

    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/physics.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/polar.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/cartesian.cxx
)

add_library(cparticles STATIC ${SOURCE_FILES})
target_include_directories(cparticles PRIVATE include/templates)
target_include_directories(cparticles PUBLIC include)
target_link_libraries(cparticles LINK_PUBLIC cstructs ctools cio)
target_compile_options(cparticles PRIVATE -fPIC)
