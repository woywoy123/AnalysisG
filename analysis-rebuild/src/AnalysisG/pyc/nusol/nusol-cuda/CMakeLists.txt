set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/nusol-cuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/nusol.cu
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/nusol.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/torch.cu
)

add_library(nusol-cuda STATIC ${SOURCE_FILES})
target_include_directories(nusol-cuda PRIVATE include/nusol)
target_include_directories(nusol-cuda PUBLIC include)
target_link_libraries(nusol-cuda LINK_PUBLIC 
    operators-cuda physics-cuda cartesian-cuda polar-cuda
)
