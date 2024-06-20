set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/generators/analysis.h)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/analysis.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/event_build.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_build.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizer_build.cxx
)

add_library(canalysis STATIC ${SOURCE_FILES})
target_include_directories(canalysis PRIVATE include/generators)
target_include_directories(canalysis PUBLIC include)
target_link_libraries(canalysis LINK_PUBLIC csampletracer cstructs cdataloader coptimizer)
target_compile_options(canalysis PRIVATE -fPIC)

