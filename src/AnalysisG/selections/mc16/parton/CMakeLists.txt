set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/parton.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/parton.cxx)

add_library(cparton STATIC ${SOURCE_FILES})
target_include_directories(cparton PUBLIC ../parton)
target_link_libraries(cparton  PUBLIC cselection_template cevent_bsm_4tops)
target_compile_options(cparton PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/mc16/parton" "selections/parton" parton "")
