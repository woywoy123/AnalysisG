set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/nusol-cuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/nusol.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/nusol.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/torch.cu
)

add_library(nusol-cuda STATIC ${SOURCE_FILES})
target_include_directories(nusol-cuda PRIVATE include/nusol)
target_include_directories(nusol-cuda PUBLIC include)
target_link_libraries(nusol-cuda PUBLIC operators-cuda physics-cuda cartesian-cuda polar-cuda)
target_compile_options(nusol-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
set_property(TARGET nusol-cuda PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(nusol-cuda PRIVATE -fPIC)




