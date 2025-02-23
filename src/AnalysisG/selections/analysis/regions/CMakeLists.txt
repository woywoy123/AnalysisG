
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/regions.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/regions.cxx)

add_library(cregions STATIC ${SOURCE_FILES})
target_include_directories(cregions PUBLIC ../regions)
target_link_libraries(cregions  PUBLIC cselection_template cevent_ssml_mc20)
target_compile_options(cregions PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/analysis/regions" "selections/regions" regions "")
