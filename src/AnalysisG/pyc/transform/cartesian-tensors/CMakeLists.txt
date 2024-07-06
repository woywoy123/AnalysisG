set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/transform/cartesian.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cartesian.cxx)

add_library(cartesian-tensors STATIC ${SOURCE_FILES})
target_include_directories(cartesian-tensors PRIVATE include/transform)
target_include_directories(cartesian-tensors PUBLIC include)
target_link_libraries(cartesian-tensors LINK_PUBLIC ${TORCH_LIBRARIES})
