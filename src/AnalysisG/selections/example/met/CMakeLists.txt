set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/met.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/met.cxx)

add_library(cmet STATIC ${SOURCE_FILES})
target_include_directories(cmet PUBLIC ../met)
target_link_libraries(cmet  PUBLIC cselection_template cevent_bsm_4tops)
target_compile_options(cmet PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/example/met" "selections/example" met "")
