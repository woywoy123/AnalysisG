set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topefficiency.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topefficiency.cxx)

add_library(ctopefficiency STATIC ${SOURCE_FILES})
target_include_directories(ctopefficiency PUBLIC ../topefficiency)
target_link_libraries(ctopefficiency  PUBLIC cselection_template cevent_bsm_4tops cevent_gnn)
target_compile_options(ctopefficiency PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/performance/topefficiency" "selections/performance" topefficiency "")
