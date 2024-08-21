set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topjets.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topjets.cxx)

add_library(ctopjets STATIC ${SOURCE_FILES})
target_include_directories(ctopjets PUBLIC ../topjets)
target_link_libraries(ctopjets  PUBLIC cselection_template cevent_bsm_4tops)
target_compile_options(ctopjets PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/mc16/topjets" "selections/mc16" topjets "")
