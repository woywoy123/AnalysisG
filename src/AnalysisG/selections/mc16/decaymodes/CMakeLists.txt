set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/decaymodes.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/decaymodes.cxx)

add_library(cdecaymodes STATIC ${SOURCE_FILES})
target_include_directories(cdecaymodes PUBLIC ../decaymodes)
target_link_libraries(cdecaymodes  PUBLIC cselection_template cevent_bsm_4tops)
target_compile_options(cdecaymodes PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/mc16/decaymodes" "selections/mc16" decaymodes "")


