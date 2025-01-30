
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/event.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/event.cxx)

add_library(cevent STATIC ${SOURCE_FILES})
target_include_directories(cevent PUBLIC ../event)
target_link_libraries(cevent  PUBLIC cselection_template cevent_bsm_4tops)
target_compile_options(cevent PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/mc16/event" "selections/event" event "")
