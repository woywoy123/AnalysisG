if(CMAKE_ANALYSISG_CUDA)
set(SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/include/graph/graph.cuh
    ${CMAKE_CURRENT_LIST_DIR}/include/graph/base.cuh
    ${CMAKE_CURRENT_LIST_DIR}/graph.cu
)
add_library(pyc_graph ${SOURCE_FILES})
set_property(TARGET pyc_graph PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(pyc_graph PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
else()
set(SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/include/graph/graph.h
    ${CMAKE_CURRENT_LIST_DIR}/graph.cxx
)
add_library(pyc_graph ${SOURCE_FILES})
endif()

target_include_directories(pyc_graph PRIVATE include/graph)
target_include_directories(pyc_graph PUBLIC include)
target_link_libraries(pyc_graph PUBLIC pyc_utils pyc_physics pyc_operators)
target_compile_options(pyc_graph PRIVATE -fPIC)

