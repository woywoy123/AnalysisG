set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/validation.cxx

)

add_library(cvalidation STATIC ${SOURCE_FILES})
target_include_directories(cvalidation PUBLIC ../validation)
target_link_libraries(cvalidation  PUBLIC cselection_template cevent_bsm_4tops cpyc)

if(CMAKE_ANALYSISG_CUDA)
target_link_libraries(cvalidation  PUBLIC cpyc)
endif()

target_compile_options(cvalidation PRIVATE -fPIC)
cmake_language(CALL cybuild "selections/neutrino/validation" "selections/validation" validation "")
