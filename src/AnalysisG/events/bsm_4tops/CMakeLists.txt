# --------------- DEFINE THE PARTICLES ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/particles.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/particles.cxx)

add_library(cparticle_bsm_4tops                STATIC ${SOURCE_FILES})
target_include_directories(cparticle_bsm_4tops PRIVATE include/bsm_4tops)
target_include_directories(cparticle_bsm_4tops PUBLIC include)

# --------------- DEFINE THE EVENT ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/bsm_4tops/event.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/event.cxx)

add_library(cevent_bsm_4tops                STATIC ${SOURCE_FILES})
target_include_directories(cevent_bsm_4tops PRIVATE include/bsm_4tops)
target_include_directories(cevent_bsm_4tops PUBLIC include)

target_link_libraries(cparticle_bsm_4tops   PUBLIC cparticle_template)
target_link_libraries(cevent_bsm_4tops      PUBLIC cparticle_bsm_4tops cevent_template)

target_compile_options(cparticle_bsm_4tops  PRIVATE -fPIC)
target_compile_options(cevent_bsm_4tops     PRIVATE -fPIC)

# -------- building ------- #
cmake_language(CALL cybuild "events/bsm_4tops" "events/bsm_4tops" particle_bsm_4tops "")
cmake_language(CALL cybuild "events/bsm_4tops" "events/bsm_4tops" event_bsm_4tops    "")

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )
