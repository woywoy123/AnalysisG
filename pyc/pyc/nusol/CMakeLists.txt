set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/device.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/utils.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/base.cuh

    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/nu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/nunu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/intersection.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/matrix.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/base.cu
)

add_library(cunusol-base STATIC ${SOURCE_FILES})
target_include_directories(cunusol-base PRIVATE include/nusol)
target_include_directories(cunusol-base PUBLIC include)
target_link_libraries(cunusol-base PUBLIC cuatomic cutils cuphysics cuoperators)
set_property(TARGET cunusol-base PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cunusol-base PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>
)
target_compile_options(cunusol-base PRIVATE -fPIC)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/nusol.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/nusol.cu
)

add_library(cunusol STATIC ${SOURCE_FILES})
target_include_directories(cunusol PRIVATE include/nusol)
target_include_directories(cunusol PUBLIC include)
target_link_libraries(cunusol PUBLIC cunusol-base)
set_property(TARGET cunusol PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cunusol PUBLIC 
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>
)
target_compile_options(cunusol PRIVATE -fPIC)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/nusol.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/nusol.cxx
)

add_library(cnusol ${SOURCE_FILES})
target_include_directories(cnusol PRIVATE include/nusol)
target_include_directories(cnusol PUBLIC include)
target_link_libraries(cnusol PUBLIC ctils cphysics coperators ctransform)
target_compile_options(cnusol PRIVATE -fPIC)

