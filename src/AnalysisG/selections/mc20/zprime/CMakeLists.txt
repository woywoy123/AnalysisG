set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/zprime.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/zprime.cxx)

add_library(czprime_mc20 STATIC ${SOURCE_FILES})
target_include_directories(czprime_mc20 PUBLIC ../zprime)
target_link_libraries(czprime_mc20  PUBLIC cselection_template cevent_ssml_mc20)
target_compile_options(czprime_mc20 PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/mc20/zprime" "selections/mc20" zprime_mc20 "")
