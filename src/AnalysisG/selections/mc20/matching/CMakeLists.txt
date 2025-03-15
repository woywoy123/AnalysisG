set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/matching.h)
set(SOURCE_FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/matching.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/mc20_exp.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/mc20_ref.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/mc16_ref.cxx
)

add_library(cmatching STATIC ${SOURCE_FILES})
target_include_directories(cmatching PUBLIC ../matching)
target_link_libraries( cmatching     PUBLIC cselection_template cevent_ssml_mc20 cevent_exp_mc20 cevent_bsm_4tops)
target_compile_options(cmatching     PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/mc20/matching" "selections/mc20" matching "")
