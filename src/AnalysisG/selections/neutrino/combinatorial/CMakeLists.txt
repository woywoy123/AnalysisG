
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/combinatorial.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/combinatorial.cxx)

add_library(ccombinatorial STATIC ${SOURCE_FILES})
target_include_directories(ccombinatorial PUBLIC ../combinatorial)
target_link_libraries(ccombinatorial  PUBLIC cselection_template cevent_bsm_4tops cpyc)
target_compile_options(ccombinatorial PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/neutrino/combinatorial" "selections/combinatorial" combinatorial "")
