
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/mc20_cur.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/mc20_cur.cxx)

add_library(cmc20_cur STATIC ${SOURCE_FILES})
target_include_directories(cmc20_cur PUBLIC ../topmatching_current)
target_link_libraries( cmc20_cur     PUBLIC cselection_template cevent_ssml_mc20)
target_compile_options(cmc20_cur     PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/mc20_compare/topmatching_current" "selections/mc20" mc20_cur "")
