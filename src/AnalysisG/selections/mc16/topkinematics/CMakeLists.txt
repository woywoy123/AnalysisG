set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topkinematics.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topkinematics.cxx)

add_library(ctopkinematics STATIC ${SOURCE_FILES})
target_include_directories(ctopkinematics PUBLIC ../topkinematics)
target_link_libraries(ctopkinematics  PUBLIC cselection_template cevent_bsm_4tops)
target_compile_options(ctopkinematics PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/mc16/topkinematics" "selections/mc16" topkinematics "")

