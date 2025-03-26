set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pyc/cupyc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/interface.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/operators.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/transform.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/graph.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/nusol.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/physics.cxx
)

add_library(cupyc SHARED ${SOURCE_FILES})
target_include_directories(cupyc PRIVATE include/pyc)
target_include_directories(cupyc PUBLIC include)
target_link_libraries(cupyc PUBLIC cutils cutransform cuphysics cuoperators cunusol cugraph)
set_property(TARGET cupyc PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cupyc PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_compile_options(cupyc PRIVATE -fPIC)
install(TARGETS cupyc DESTINATION .)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pyc/tpyc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/interface.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/operators.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/transform.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/graph.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/nusol.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/physics.cxx
)

add_library(tpyc SHARED ${SOURCE_FILES})
target_include_directories(tpyc PRIVATE include/pyc)
target_include_directories(tpyc PUBLIC include)
target_link_libraries(tpyc PUBLIC ctils ctransform cphysics cgraph cnusol coperators)
target_compile_options(tpyc PRIVATE -fPIC)
install(TARGETS tpyc DESTINATION .)
