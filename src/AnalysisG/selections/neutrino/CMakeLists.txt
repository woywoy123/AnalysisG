set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/neutrino.h
    ${CMAKE_CURRENT_SOURCE_DIR}/neutrino.cxx
)

add_library(cneutrino STATIC ${SOURCE_FILES})
target_include_directories(cneutrino PUBLIC ../neutrino)
target_link_libraries(cneutrino  PUBLIC cselection_template cevent_bsm_4tops nusol-cuda)
target_compile_options(cneutrino PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/neutrino" "selections/neutrino" neutrino "")

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

