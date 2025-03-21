set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/AnalysisG/analysis.h)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/analysis.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/event_build.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_build.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/selection_build.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizer_build.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/inference_build.cxx
)

add_library(canalysis STATIC ${SOURCE_FILES})
target_include_directories(canalysis PRIVATE include/AnalysisG)
target_include_directories(canalysis PUBLIC include)
target_link_libraries(canalysis PUBLIC 
    csampletracer cstructs cdataloader 
    coptimizer cselection_template cio
    ROOT::RIO ${TORCH_LIBRARIES}
)
target_compile_options(canalysis PRIVATE -fPIC)
target_compile_options(canalysis PRIVATE -Wall -pedantic -Wextra)

