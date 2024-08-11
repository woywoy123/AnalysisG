set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/toptruthjets.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/toptruthjets.cxx)

add_library(ctoptruthjets STATIC ${SOURCE_FILES})
target_include_directories(ctoptruthjets PUBLIC ../toptruthjets)
target_link_libraries(ctoptruthjets  PUBLIC cselection_template cevent_bsm_4tops)
target_compile_options(ctoptruthjets PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/mc16/toptruthjets" "selections/mc16" toptruthjets "")
