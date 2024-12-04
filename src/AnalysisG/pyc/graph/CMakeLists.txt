set(SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/include/graph/graph.cuh
    ${CMAKE_CURRENT_LIST_DIR}/include/graph/base.cuh
    ${CMAKE_CURRENT_LIST_DIR}/graph.cu
)

add_library(cugraph STATIC ${SOURCE_FILES})
target_include_directories(cugraph PRIVATE include/graph)
target_include_directories(cugraph PUBLIC include)
target_link_libraries(cugraph PUBLIC cuatomic cutils cuphysics cuoperators)
set_property(TARGET cugraph PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cugraph PUBLIC 
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>
)
target_compile_options(cugraph PRIVATE -fPIC)

set(SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/include/graph/graph.h
    ${CMAKE_CURRENT_LIST_DIR}/graph.cxx
)

add_library(cgraph ${SOURCE_FILES})
target_include_directories(cgraph PRIVATE include/graph)
target_include_directories(cgraph PUBLIC include)
target_link_libraries(cgraph PUBLIC ctils)
target_compile_options(cgraph PRIVATE -fPIC)



