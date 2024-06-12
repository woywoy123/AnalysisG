add_subdirectory(bsm_4tops)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops/event.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops/particles.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops/event.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops/particles.cxx
)

add_library(cevent_bsm_4tops STATIC ${SOURCE_FILES})
target_include_directories(cevent_bsm_4tops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cevent_bsm_4tops LINK_PUBLIC cparticles cevent)
target_compile_options(cevent_bsm_4tops PRIVATE -fPIC)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.cpp
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/event_bsm_4tops.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/event_bsm_4tops.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        ${CMAKE_CURRENT_SOURCE_DIR}/event_bsm_4tops.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.cpp
)

python_add_library(event_bsm_4tops MODULE ${CMAKE_CURRENT_SOURCE_DIR}/bsm_4tops.cpp WITH_SOABI)
target_link_libraries(event_bsm_4tops LINK_PUBLIC cevent_bsm_4tops)

install(FILES __init__.pxd DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/events)
install(TARGETS event_bsm_4tops  DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/events)

