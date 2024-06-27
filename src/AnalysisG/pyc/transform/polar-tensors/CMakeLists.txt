set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/transform/polar.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/polar.cxx)

add_library(polar-tensors STATIC ${SOURCE_FILES})
target_include_directories(polar-tensors PRIVATE include/transform)
target_include_directories(polar-tensors PUBLIC include)
target_link_libraries(polar-tensors LINK_PUBLIC ${TORCH_LIBRARIES})

