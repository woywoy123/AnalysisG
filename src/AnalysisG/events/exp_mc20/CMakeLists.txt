# --------------- DEFINE THE PARTICLES ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/exp_mc20/particles.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/particles.cxx)

add_library(cparticle_exp_mc20                STATIC ${SOURCE_FILES})
target_include_directories(cparticle_exp_mc20 PRIVATE include/exp_mc20)
target_include_directories(cparticle_exp_mc20 PUBLIC include)

# --------------- DEFINE THE EVENT ------------------ #
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/exp_mc20/event.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/event.cxx)

add_library(cevent_exp_mc20                STATIC ${SOURCE_FILES})
target_include_directories(cevent_exp_mc20 PRIVATE include/exp_mc20)
target_include_directories(cevent_exp_mc20 PUBLIC include)

target_link_libraries(cparticle_exp_mc20   LINK_PUBLIC cparticle_template)
target_link_libraries(cevent_exp_mc20      LINK_PUBLIC cparticle_exp_mc20 cevent_template)

target_compile_options(cparticle_exp_mc20  PRIVATE -fPIC)
target_compile_options(cevent_exp_mc20     PRIVATE -fPIC)

# -------------- building -------------- #
cmake_language(CALL cybuild "events/exp_mc20" "events/exp_mc20" particle_exp_mc20 "")
cmake_language(CALL cybuild "events/exp_mc20" "events/exp_mc20" event_exp_mc20    "")

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )
