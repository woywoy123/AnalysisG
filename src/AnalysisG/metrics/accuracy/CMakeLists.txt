set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/accuracy.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/collector.cxx
)
add_library(cmetric_accuracy STATIC ${SOURCE_FILES})
target_include_directories(cmetric_accuracy PRIVATE include/metrics)
target_include_directories(cmetric_accuracy PUBLIC include)
target_link_libraries(cmetric_accuracy PUBLIC cmetric_template croc)
target_compile_options(cmetric_accuracy PRIVATE -fPIC)
target_compile_options(cmetric_accuracy PRIVATE -Wall -pedantic -Wextra)
cmake_language(CALL cybuild "metrics/accuracy" "metrics" metric_accuracy "")
