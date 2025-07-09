set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/met.h)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/met.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/misc.cxx
)

add_library(cmissing_et STATIC ${SOURCE_FILES})
target_include_directories(cmissing_et PUBLIC ../met)
target_link_libraries(cmissing_et  PUBLIC cselection_template cevent_bsm_4tops cpyc)
target_compile_options(cmissing_et PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/mc16/met" "selections/met" missing_et "")
