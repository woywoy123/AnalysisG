set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/validation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/validation.cxx
)

add_library(cvalidation STATIC ${SOURCE_FILES})
target_include_directories(cvalidation PUBLIC ../validation)

target_link_libraries(cvalidation  PUBLIC cselection_template cevent_bsm_4tops cpyc)
if(CMAKE_ANALYSISG_CUDA)
target_compile_definitions(cvalidation PRIVATE PYC_CUDA ON)
endif()
target_compile_options(cvalidation PRIVATE -fPIC)

cmake_language(CALL cybuild "selections/neutrino/validation" "selections/neutrino/validation" validation "")
file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

