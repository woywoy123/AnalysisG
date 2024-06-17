set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/physics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/polar.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/physics/cartesian.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/physics.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/polar.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cartesian.cxx
)

add_library(physics-tensors STATIC ${SOURCE_FILES})
target_include_directories(physics-tensors PRIVATE include/physics)
target_include_directories(physics-tensors PUBLIC include)
target_link_libraries(physics-tensors LINK_PUBLIC polar-tensors cartesian-tensors)
