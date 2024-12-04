set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pyc/cupyc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/interface.cxx
)

add_library(cupyc SHARED ${SOURCE_FILES})
target_include_directories(cupyc PRIVATE include/pyc)
target_include_directories(cupyc PUBLIC include)
target_link_libraries(cupyc PUBLIC cutils cutransform cuphysics cuoperators cunusol cugraph)

set_property(TARGET cupyc PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cupyc PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_compile_options(cupyc PRIVATE -fPIC)

set(SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/include/pyc/tpyc.h
    ${CMAKE_CURRENT_LIST_DIR}/tensor/interface.cxx
)

add_library(tpyc ${SOURCE_FILES})
target_include_directories(tpyc PRIVATE include/pyc)
target_include_directories(tpyc PUBLIC include)
target_link_libraries(tpyc PUBLIC ctils ctransform cphysics cgraph cnusol coperators)
target_compile_options(tpyc PRIVATE -fPIC)
