set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/operators/operators.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/operators.cxx)

add_library(operators-tensor STATIC ${SOURCE_FILES})
target_include_directories(operators-tensor PRIVATE include/operators)
target_include_directories(operators-tensor PUBLIC include)
target_link_libraries(operators-tensor LINK_PUBLIC ${TORCH_LIBRARIES})
