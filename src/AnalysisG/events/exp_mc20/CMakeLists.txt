# --------------- DEFINE THE PARTICLES ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/exp_mc20/particles.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/particles.cxx)

add_library(cparticle_exp_mc20 STATIC ${SOURCE_FILES})
target_include_directories(cparticle_exp_mc20 PUBLIC include)
target_include_directories(cparticle_exp_mc20 PRIVATE include/exp_mc20)
target_link_libraries(cparticle_exp_mc20      LINK_PUBLIC cparticles)
target_compile_options(cparticle_exp_mc20     PRIVATE -fPIC)


# --------------- DEFINE THE EVENT ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/exp_mc20/event.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/event.cxx)

add_library(cevent_exp_mc20 STATIC ${SOURCE_FILES})
target_include_directories(cevent_exp_mc20 PUBLIC include)
target_include_directories(cevent_exp_mc20 PRIVATE include/exp_mc20)
target_link_libraries(cevent_exp_mc20      LINK_PUBLIC cparticle_exp_mc20 cevent)
target_compile_options(cevent_exp_mc20     PRIVATE -fPIC)

# -------------- building -------------- #
add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/particle_exp_mc20.cpp
DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_exp_mc20.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_exp_mc20.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_exp_mc20.pyx
    --output-file ${CMAKE_CURRENT_SOURCE_DIR}/particle_exp_mc20.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/event_exp_mc20.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/event_exp_mc20.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/event_exp_mc20.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/event_exp_mc20.pyx 
    --output-file ${CMAKE_CURRENT_SOURCE_DIR}/event_exp_mc20.cpp
)

# ------- linking -------- #
python_add_library(event_exp_mc20    MODULE ${CMAKE_CURRENT_SOURCE_DIR}/event_exp_mc20.cpp WITH_SOABI)
python_add_library(particle_exp_mc20 MODULE ${CMAKE_CURRENT_SOURCE_DIR}/particle_exp_mc20.cpp WITH_SOABI)

target_link_libraries(event_exp_mc20    LINK_PUBLIC cevent_exp_mc20)
target_link_libraries(particle_exp_mc20 LINK_PUBLIC cparticle_exp_mc20)

install(TARGETS event_exp_mc20     DESTINATION ${SKBUILD_PROJECT_NAME}/events/exp_mc20)
install(TARGETS particle_exp_mc20  DESTINATION ${SKBUILD_PROJECT_NAME}/events/exp_mc20)

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/event_exp_mc20.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/particle.cpp)
