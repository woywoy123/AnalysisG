add_library(physics-cuda "")
target_sources(physics-cuda 
    PUBLIC 
        ${CMAKE_CURRENT_LIST_DIR}/physics.h
        ${CMAKE_CURRENT_LIST_DIR}/cartesian.h
        ${CMAKE_CURRENT_LIST_DIR}/polar.h
        ${CMAKE_CURRENT_LIST_DIR}/physics.cu
        ${CMAKE_CURRENT_LIST_DIR}/kernel.cu
        ${CMAKE_CURRENT_LIST_DIR}/torch.cu
)



target_link_libraries(physics-cuda PUBLIC ${TORCH_LIBRARIES} cartesian-cuda)
target_compile_options(physics-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_include_directories(physics-cuda PUBLIC ${CMAKE_CURRENT_LIST_DIR})

