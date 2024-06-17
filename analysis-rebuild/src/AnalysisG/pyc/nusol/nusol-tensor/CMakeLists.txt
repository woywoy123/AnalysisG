set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/nusol.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nusol/nusol-tools.h
)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/nusol.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/nusol_tools.cxx
)

add_library(nusol-tensor STATIC ${SOURCE_FILES})
target_include_directories(nusol-tensor PRIVATE include/nusol)
target_include_directories(nusol-tensor PUBLIC include)
target_link_libraries(nusol-tensor 
    LINK_PUBLIC operators-tensor polar-tensors cartesian-tensors physics-tensors
)
