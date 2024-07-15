# --------------- DEFINE THE PARTICLES ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/particles.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/particles.cxx)
add_library(cparticle_bsm_4tops STATIC ${SOURCE_FILES})
target_link_libraries(cparticle_bsm_4tops PUBLIC cparticles)
target_compile_options(cparticle_bsm_4tops PRIVATE -fPIC)
target_include_directories(cparticle_bsm_4tops PUBLIC include)
target_include_directories(cparticle_bsm_4tops PRIVATE include/bsm_4tops)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/particle_bsm_4tops.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_bsm_4tops.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_bsm_4tops.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_bsm_4tops.pyx 
    --output-file ${CMAKE_CURRENT_SOURCE_DIR}/particle_bsm_4tops.cpp
)

python_add_library(particle_bsm_4tops MODULE ${CMAKE_CURRENT_SOURCE_DIR}/particle_bsm_4tops.cpp WITH_SOABI)
target_link_libraries(particle_bsm_4tops LINK_PUBLIC cparticle_bsm_4tops)
install(FILES __init__.pxd          DESTINATION ${SKBUILD_PROJECT_NAME}/events/bsm_4tops)
install(TARGETS particle_bsm_4tops  DESTINATION ${SKBUILD_PROJECT_NAME}/events/bsm_4tops)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/particle_bsm_4tops.cpp)

# --------------- DEFINE THE EVENT ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/event.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/event.cxx)
add_library(cevent_bsm_4tops STATIC ${SOURCE_FILES})
target_link_libraries(cevent_bsm_4tops LINK_PUBLIC cparticle_bsm_4tops cevent)
target_compile_options(cevent_bsm_4tops PRIVATE -fPIC)
target_include_directories(cevent_bsm_4tops PUBLIC include)
target_include_directories(cevent_bsm_4tops PRIVATE include/bsm_4tops)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/event_bsm_4tops.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/event_bsm_4tops.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/event_bsm_4tops.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/event_bsm_4tops.pyx 
    --output-file ${CMAKE_CURRENT_SOURCE_DIR}/event_bsm_4tops.cpp
)

python_add_library(event_bsm_4tops MODULE ${CMAKE_CURRENT_SOURCE_DIR}/event_bsm_4tops.cpp WITH_SOABI)
target_link_libraries(event_bsm_4tops LINK_PUBLIC cevent_bsm_4tops)
install(FILES __init__.pxd          DESTINATION ${SKBUILD_PROJECT_NAME}/events/bsm_4tops)
install(TARGETS event_bsm_4tops     DESTINATION ${SKBUILD_PROJECT_NAME}/events/bsm_4tops)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/event_bsm_4tops.cpp)

