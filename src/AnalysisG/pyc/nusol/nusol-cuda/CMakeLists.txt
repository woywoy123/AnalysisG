set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/base.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/shape.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/base/matrix.cu
)

add_library(cubase ${SOURCE_FILES})
target_include_directories(cubase PRIVATE include/nusol kernels)
target_include_directories(cubase PUBLIC include)
target_compile_options(cubase PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_link_libraries(cubase PRIVATE ${TORCH_LIBRARIES} cutils)
set_property(TARGET cubase PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cubase PRIVATE -fPIC)



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
target_link_libraries(nusol-cuda PUBLIC operators-cuda physics-cuda cartesian-cuda polar-cuda cutils cubase)
target_compile_options(nusol-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
set_property(TARGET nusol-cuda PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(nusol-cuda PRIVATE -fPIC)

