set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/event_template.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/solvers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/nusol.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/nunu.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/mtx.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/event_template.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/properties.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/solvers.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/nusol.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/nunu.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/mtx.cxx
)

add_library(cevent_template ${SOURCE_FILES})
target_include_directories(cevent_template PRIVATE include/templates)
target_include_directories(cevent_template PUBLIC include)
target_link_libraries(cevent_template  PUBLIC cstructs cparticle_template ctools cmeta)
target_compile_options(cevent_template PRIVATE -fPIC)
target_compile_options(cevent_template PRIVATE -Wall -pedantic -Wextra -Wshadow)
