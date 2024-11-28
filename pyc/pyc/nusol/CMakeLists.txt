set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/device.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/utils.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/base.cuh

    ${CMAKE_CURRENT_SOURCE_DIR}/nu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/nunu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/intersection.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/base.cu
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
    ${CMAKE_CURRENT_SOURCE_DIR}/nusol.cu
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
