set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/particles/particles.h)
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/particles.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/interfaces.cxx

    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/physics.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/polar.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/cartesian.cxx
)

add_library(cparticles STATIC ${SOURCE_FILES})
target_include_directories(cparticles PRIVATE include/particles)
target_include_directories(cparticles PUBLIC include)
target_link_libraries(cparticles LINK_PUBLIC cstructs ctools)
target_compile_options(cparticles PRIVATE -fPIC)

add_custom_command(
OUTPUT 
    ${CMAKE_CURRENT_SOURCE_DIR}/cparticles.cxx
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particles.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particles.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/particles.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cparticles.cxx
)

python_add_library(particles MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cparticles.cxx WITH_SOABI)
target_link_libraries(particles PUBLIC cparticles cstructs)
install(TARGETS particles DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules)
install(FILES ${SOURCE_FILES} DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules/particles)
install(FILES ${HEADER_FILES} DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules/particles)


