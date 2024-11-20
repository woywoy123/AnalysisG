set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pyc/pyc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interface.cxx
)
add_library(cupyc SHARED ${SOURCE_FILES})
target_include_directories(cupyc PRIVATE include/pyc)
target_include_directories(cupyc PUBLIC include)
target_compile_options(cupyc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_link_libraries(cupyc PUBLIC cutils cutransform cuphysics cuoperators cunusol ${TORCH_LIBRARIES})
set_property(TARGET cupyc PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cupyc PRIVATE -fPIC)
install(TARGETS cupyc DESTINATION .)
