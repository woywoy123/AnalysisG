set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/physics-cuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/cartesian-cuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/polar-cuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/torch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/physics.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/physics.cxx
)

add_library(physics-cuda STATIC ${SOURCE_FILES})
target_include_directories(physics-cuda PRIVATE include/physics)
target_include_directories(physics-cuda PUBLIC include)
target_link_libraries(physics-cuda PUBLIC ${TORCH_LIBRARIES} polar-cuda cartesian-cuda)
set_property(TARGET physics-cuda PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(physics-cuda PRIVATE -fPIC)
