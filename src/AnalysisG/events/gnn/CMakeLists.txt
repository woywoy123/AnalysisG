# --------------- DEFINE THE PARTICLES ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/inference/gnn-particles.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/particles.cxx)
add_library(cparticle_gnn                STATIC ${SOURCE_FILES})
target_include_directories(cparticle_gnn PUBLIC include)
target_include_directories(cparticle_gnn PRIVATE include/inference)
target_link_libraries(cparticle_gnn      LINK_PUBLIC cparticles)
target_compile_options(cparticle_gnn     PRIVATE -fPIC)

# --------------- DEFINE THE EVENT ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/inference/gnn-event.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/event.cxx)
add_library(cevent_gnn                STATIC ${SOURCE_FILES})
target_include_directories(cevent_gnn PUBLIC include)
target_include_directories(cevent_gnn PRIVATE include/inference)
target_link_libraries(cevent_gnn      LINK_PUBLIC cparticle_gnn cevent)
target_compile_options(cevent_gnn     PRIVATE -fPIC)

# -------- building ------- #
add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/particle_gnn.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_gnn.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_gnn.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_gnn.pyx 
    --output-file ${CMAKE_CURRENT_SOURCE_DIR}/particle_gnn.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/event_gnn.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/event_gnn.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/event_gnn.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/event_gnn.pyx 
    --output-file ${CMAKE_CURRENT_SOURCE_DIR}/event_gnn.cpp
)

# -------- linking --------#
python_add_library(particle_gnn MODULE ${CMAKE_CURRENT_SOURCE_DIR}/particle_gnn.cpp WITH_SOABI)
python_add_library(event_gnn    MODULE ${CMAKE_CURRENT_SOURCE_DIR}/event_gnn.cpp WITH_SOABI)

target_link_libraries(particle_gnn LINK_PUBLIC cparticle_gnn)
target_link_libraries(event_gnn    LINK_PUBLIC cevent_gnn)

install(TARGETS particle_gnn DESTINATION ${SKBUILD_PROJECT_NAME}/events/gnn)
install(TARGETS event_gnn    DESTINATION ${SKBUILD_PROJECT_NAME}/events/gnn)

file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/particle_gnn.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/event_gnn.cpp)

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

