set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/event.h
    ${CMAKE_CURRENT_SOURCE_DIR}/particles.h
    ${CMAKE_CURRENT_SOURCE_DIR}/event.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/particles.cxx
)

add_library(cbsm_4tops STATIC ${SOURCE_FILES})
target_link_libraries(cbsm_4tops LINK_PUBLIC cparticles cevent cio)
target_compile_options(cbsm_4tops PRIVATE -fPIC)
