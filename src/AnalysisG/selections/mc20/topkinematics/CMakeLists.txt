set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topkinematics.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topkinematics.cxx)

add_library(ctopkinematics_mc20 STATIC ${SOURCE_FILES})
target_include_directories(ctopkinematics_mc20 PUBLIC ../topkinematics)
target_link_libraries(ctopkinematics_mc20 PUBLIC cselection_template cevent_ssml_mc20)
target_compile_options(ctopkinematics_mc20 PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/mc20/topkinematics" "selections/mc20" topkinematics_mc20 "")
