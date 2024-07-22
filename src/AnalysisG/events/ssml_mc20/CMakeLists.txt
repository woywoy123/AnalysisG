# --------------- DEFINE THE PARTICLES ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/ssml_mc20/particles.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/particles.cxx)
add_library(cparticle_ssml_mc20 STATIC ${SOURCE_FILES})
target_link_libraries(cparticle_ssml_mc20 LINK_PUBLIC cparticle)
target_compile_options(cparticle_ssml_mc20 PRIVATE -fPIC)
target_include_directories(cparticle_ssml_mc20 PUBLIC include)
target_include_directories(cparticle_ssml_mc20 PRIVATE include/ssml_mc20)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/particle_ssml_mc20.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_ssml_mc20.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_ssml_mc20.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_ssml_mc20.pyx 
    --output-file ${CMAKE_CURRENT_SOURCE_DIR}/particle_ssml_mc20.cpp
)

python_add_library(particle_ssml_mc20 MODULE ${CMAKE_CURRENT_SOURCE_DIR}/particle_ssml_mc20.cpp WITH_SOABI)
target_link_libraries(particle_ssml_mc20 LINK_PUBLIC cparticle_ssml_mc20)
install(FILES __init__.pxd          DESTINATION ${SKBUILD_PROJECT_NAME}/events)
install(TARGETS particle_ssml_mc20  DESTINATION ${SKBUILD_PROJECT_NAME}/events)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/particle_ssml_mc20.cpp)


# --------------- DEFINE THE EVENT ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/ssml_mc20/event.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/event.cxx)
add_library(cevent_ssml_mc20 STATIC ${SOURCE_FILES})
target_link_libraries(cevent_ssml_mc20 LINK_PUBLIC cparticle_ssml_mc20 cevent)
target_compile_options(cevent_ssml_mc20 PRIVATE -fPIC)
target_include_directories(cevent_ssml_mc20 PUBLIC include)
target_include_directories(cevent_ssml_mc20 PRIVATE include/ssml_mc20)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/event_ssml_mc20.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/event_ssml_mc20.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/event_ssml_mc20.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/event_ssml_mc20.pyx 
    --output-file ${CMAKE_CURRENT_SOURCE_DIR}/event_ssml_mc20.cpp
)

python_add_library(event_ssml_mc20    MODULE ${CMAKE_CURRENT_SOURCE_DIR}/event_ssml_mc20.cpp WITH_SOABI)
target_link_libraries(event_ssml_mc20 LINK_PUBLIC cevent_ssml_mc20)
install(FILES __init__.pxd            DESTINATION ${SKBUILD_PROJECT_NAME}/events/)
install(TARGETS event_ssml_mc20       DESTINATION ${SKBUILD_PROJECT_NAME}/events/)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/event_ssml_mc20.cpp)

