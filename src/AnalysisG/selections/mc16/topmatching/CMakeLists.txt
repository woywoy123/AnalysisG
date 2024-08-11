set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topmatching.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topmatching.cxx)

add_library(ctopmatching STATIC ${SOURCE_FILES})
target_include_directories(ctopmatching PUBLIC ../topmatching)
target_link_libraries(ctopmatching  PUBLIC cselection_template cevent_bsm_4tops)
target_compile_options(ctopmatching PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/mc16/topmatching" "selections/mc16" topmatching "")
