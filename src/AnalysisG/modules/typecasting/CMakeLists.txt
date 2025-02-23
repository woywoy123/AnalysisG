set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/tensor_cast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/merge_cast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/vector_cast.h
)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/typecasting.cxx)
add_library(ctypecasting ${SOURCE_FILES})
target_include_directories(ctypecasting PRIVATE include/tools)
target_include_directories(ctypecasting PUBLIC include)
target_link_libraries(ctypecasting PUBLIC ${TORCH_LIBRARIES} ${ROOT_LIBRARIES})
target_compile_options(ctypecasting PRIVATE -fPIC)
target_compile_options(ctypecasting PUBLIC -Wall -pedantic -Wextra)

