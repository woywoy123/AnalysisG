set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/childrenkinematics.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/childrenkinematics.cxx)

add_library(cchildrenkinematics STATIC ${SOURCE_FILES})
target_include_directories(cchildrenkinematics PUBLIC ../childrenkinematics)
target_link_libraries(cchildrenkinematics      PUBLIC cselection_template cevent_bsm_4tops)
target_compile_options(cchildrenkinematics     PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/mc16/childrenkinematics" "selections/mc16" childrenkinematics "")
