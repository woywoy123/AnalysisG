set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/metrics/metrics.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/metrics.cxx)

add_library(cmetrics STATIC ${SOURCE_FILES})
target_include_directories(cmetrics PRIVATE include/metrics)
target_include_directories(cmetrics PUBLIC include)
target_link_libraries(cmetrics LINK_PUBLIC cmodel graph-cuda)
target_compile_options(cmetrics PRIVATE -fPIC)
