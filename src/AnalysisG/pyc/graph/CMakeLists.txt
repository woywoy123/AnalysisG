// Check if CUDA is enabled for AnalysisG
if(CMAKE_ANALYSISG_CUDA)
// Define the source files for CUDA compilation
set(SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/include/graph/graph.cuh // CUDA header for graph operations
    ${CMAKE_CURRENT_LIST_DIR}/include/graph/base.cuh // CUDA header for base graph utilities
    ${CMAKE_CURRENT_LIST_DIR}/include/graph/pagerank.cuh // CUDA header for PageRank algorithm
    ${CMAKE_CURRENT_LIST_DIR}/pagerank.cu // CUDA source for PageRank algorithm
    ${CMAKE_CURRENT_LIST_DIR}/graph.cu // CUDA source for graph operations
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction.cu // CUDA source for graph reconstruction
)
// Add a library target named pyc_graph using the defined source files
add_library(pyc_graph ${SOURCE_FILES})
// Enable separate compilation for the pyc_graph CUDA target
set_property(TARGET pyc_graph PROPERTY CUDA_SEPARATE_COMPILATION ON)
// Set compile options for the pyc_graph target, specifically enabling experimental relaxed constexpr for CUDA (public)
target_compile_options(pyc_graph PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
else() // If CUDA is not enabled
// Define the source files for C++ compilation
set(SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/include/graph/graph.h // C++ header for graph operations
    ${CMAKE_CURRENT_LIST_DIR}/graph.cxx // C++ source for graph operations
)
// Add a library target named pyc_graph using the defined source files
add_library(pyc_graph ${SOURCE_FILES})
endif() // End of the CUDA check

// Add include directories for the pyc_graph target (private)
target_include_directories(pyc_graph PRIVATE include/graph)
// Add include directories for the pyc_graph target (public)
target_include_directories(pyc_graph PUBLIC include)
// Link the pyc_graph target with utility, physics, and operator libraries
target_link_libraries(pyc_graph PUBLIC pyc_utils pyc_physics pyc_operators)
// Set compile options for the pyc_graph target, enabling Position Independent Code (PIC)
target_compile_options(pyc_graph PRIVATE -fPIC)

