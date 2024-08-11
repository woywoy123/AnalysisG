# --------------- DEFINE THE PARTICLES ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/inference/gnn-particles.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/particles.cxx)

add_library(cparticle_gnn                STATIC ${SOURCE_FILES})
target_include_directories(cparticle_gnn PRIVATE include/inference)
target_include_directories(cparticle_gnn PUBLIC include)


# --------------- DEFINE THE EVENT ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/inference/gnn-event.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/event.cxx)

add_library(cevent_gnn                STATIC ${SOURCE_FILES})
target_include_directories(cevent_gnn PRIVATE include/inference)
target_include_directories(cevent_gnn PUBLIC include)

target_link_libraries(cparticle_gnn   PUBLIC cparticle_template)
target_link_libraries(cevent_gnn      PUBLIC cparticle_gnn cevent_template)

target_compile_options(cparticle_gnn  PRIVATE -fPIC)
target_compile_options(cevent_gnn     PRIVATE -fPIC)

# -------- building ------- #
cmake_language(CALL cybuild "events/gnn" "events/gnn" particle_gnn "")
cmake_language(CALL cybuild "events/gnn" "events/gnn" event_gnn    "")

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

