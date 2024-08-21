set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/zprime.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/zprime.cxx)

add_library(czprime STATIC ${SOURCE_FILES})
target_include_directories(czprime PUBLIC ../zprime)
target_link_libraries(czprime  PUBLIC cselection_template cevent_bsm_4tops)
target_compile_options(czprime PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/mc16/zprime" "selections/mc16" zprime "")
