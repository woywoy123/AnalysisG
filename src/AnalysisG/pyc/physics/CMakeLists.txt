if(CMAKE_ANALYSISG_CUDA)
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/physics.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/base.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/physics.cu
)

add_library(cuphysics ${SOURCE_FILES})
target_include_directories(cuphysics PRIVATE include/physics)
target_include_directories(cuphysics PUBLIC include)
target_compile_options(cuphysics PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_link_libraries(cuphysics PUBLIC cutils cutransform cuatomic)
set_property(TARGET cuphysics PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cuphysics PRIVATE -fPIC)
endif()

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/physics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/physics.cxx
)

add_library(cphysics ${SOURCE_FILES})
target_include_directories(cphysics PRIVATE include/physics)
target_include_directories(cphysics PUBLIC include)
target_link_libraries(cphysics PUBLIC ctils)
target_compile_options(cphysics PRIVATE -fPIC)

