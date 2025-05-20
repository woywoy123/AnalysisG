set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/pagerank.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/collector.cxx
)

add_library(cmetric_pagerank STATIC ${SOURCE_FILES})
target_include_directories(cmetric_pagerank PRIVATE include/metrics)
target_include_directories(cmetric_pagerank PUBLIC include)
target_link_libraries(cmetric_pagerank PUBLIC cmetric_template)
target_compile_options(cmetric_pagerank PRIVATE -fPIC)
target_compile_options(cmetric_pagerank PRIVATE -Wall -pedantic -Wextra)
cmake_language(CALL cybuild "metrics/pagerank" "metrics" metric_pagerank "")
