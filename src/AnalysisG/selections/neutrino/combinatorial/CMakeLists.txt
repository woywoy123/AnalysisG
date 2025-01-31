set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/combinatorial.h
    ${CMAKE_CURRENT_SOURCE_DIR}/combinatorial.cxx
)

add_library(ccombinatorial STATIC ${SOURCE_FILES})
target_include_directories(ccombinatorial PUBLIC ../combinatorial)
target_link_libraries(ccombinatorial  PUBLIC cselection_template cevent_bsm_4tops cupyc)
target_compile_options(ccombinatorial PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/neutrino/combinatorial" "selections/neutrino/combinatorial" combinatorial "")

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

