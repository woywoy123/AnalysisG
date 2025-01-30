set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topmatching.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topmatching.cxx)

add_library(ctopmatching_mc20 STATIC ${SOURCE_FILES})
target_include_directories(ctopmatching_mc20 PUBLIC ../topmatching)
target_link_libraries(ctopmatching_mc20  PUBLIC cselection_template cevent_ssml_mc20)
target_compile_options(ctopmatching_mc20 PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/mc20/topmatching" "selections/mc20" topmatching_mc20 "")
