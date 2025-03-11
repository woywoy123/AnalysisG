if(CMAKE_ANALYSISG_CUDA)
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/atomic.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/atomic.cu

    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/utils.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cu
)
add_library(pyc_utils ${SOURCE_FILES})
target_compile_options(pyc_utils PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
set_property(TARGET pyc_utils PROPERTY CUDA_SEPARATE_COMPILATION ON)
else()
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cxx
)
add_library(pyc_utils ${SOURCE_FILES})
endif()

target_include_directories(pyc_utils PRIVATE include/cutils)
target_include_directories(pyc_utils PUBLIC include)
target_link_libraries(pyc_utils PUBLIC ${TORCH_LIBRARIES})
target_compile_options(pyc_utils PRIVATE -fPIC)
