set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/conuix.h
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/cache.h
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/matrix.h
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/conuix.h
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/memory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/solvers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/multisol.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/conuix.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/cache.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/matrix.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/conuix.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/misc.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/memory.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/solvers.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/multisol/multisol.cxx
)

add_library(cconuix STATIC ${SOURCE_FILES})
target_include_directories(cconuix PUBLIC ../conuix)
target_link_libraries(cconuix  PUBLIC cselection_template cevent_bsm_4tops cparticle_template)
target_compile_options(cconuix PRIVATE -fPIC)

cmake_language(CALL cysub_build "selections/neutrino/conuix" "selections/conuix" conuix)













