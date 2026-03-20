add_subdirectory(conuic)

set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/multisol.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/multisol.cxx)

add_library(cmultisol STATIC ${SOURCE_FILES})
target_include_directories(cmultisol PUBLIC ../multisol)
target_link_libraries(cmultisol  PUBLIC cselection_template cevent_bsm_4tops cparticle_template)
target_link_libraries(cmultisol  PUBLIC cxnunix)
target_compile_options(cmultisol PRIVATE -fPIC)

cmake_language(CALL cysub_build "selections/neutrino/multisol" "selections/multisol" multisol)


