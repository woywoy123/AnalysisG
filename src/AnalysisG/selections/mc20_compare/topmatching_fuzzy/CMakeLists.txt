
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/mc20_fuzzy.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/mc20_fuzzy.cxx)

add_library(cmc20_fuzzy STATIC ${SOURCE_FILES})
target_include_directories(cmc20_fuzzy PUBLIC ../topmatching_fuzzy)
target_link_libraries( cmc20_fuzzy     PUBLIC cselection_template cevent_exp_mc20)
target_compile_options(cmc20_fuzzy     PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/mc20_compare/topmatching_fuzzy" "selections/mc20" mc20_fuzzy "")
