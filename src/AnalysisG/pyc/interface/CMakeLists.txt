set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/interface.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/transform.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/operators.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/physics.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/nusol.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/graph.cxx
)

add_library(cpyc SHARED ${SOURCE_FILES})
target_include_directories(cpyc PRIVATE include/pyc)
target_include_directories(cpyc PUBLIC include)
target_link_libraries(cpyc PUBLIC 
    pyc_utils pyc_transform pyc_physics 
    pyc_operators pyc_nusol pyc_graph 
    cparticle_template ctypecasting
)

if (CMAKE_ANALYSISG_CUDA)
set_property(TARGET cpyc PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cpyc PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_compile_definitions(cpyc PRIVATE PYC_CUDA ON)
endif()
target_compile_options(cpyc PRIVATE -fPIC)
install(TARGETS cpyc DESTINATION .)
