# Define the source files for the cpyc shared library
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/interface.cxx # Main interface file for the library
    ${CMAKE_CURRENT_SOURCE_DIR}/transform.cxx # Source file for transformation operations
    ${CMAKE_CURRENT_SOURCE_DIR}/operators.cxx # Source file for mathematical operators
    ${CMAKE_CURRENT_SOURCE_DIR}/physics.cxx # Source file for physics calculations
    ${CMAKE_CURRENT_SOURCE_DIR}/nusol.cxx # Source file for neutrino reconstruction (NuSol)
    ${CMAKE_CURRENT_SOURCE_DIR}/graph.cxx # Source file for graph operations
)

# Add a shared library target named cpyc using the defined source files
add_library(cpyc SHARED ${SOURCE_FILES})
# Add private include directories for the cpyc target
target_include_directories(cpyc PRIVATE include/pyc)
# Add public include directories for the cpyc target
target_include_directories(cpyc PUBLIC include)
# Link the cpyc target with other pre-compiled libraries (utils, transform, physics, operators, nusol, graph, particle_template, typecasting)
target_link_libraries(cpyc PUBLIC
    pyc_utils pyc_transform pyc_physics
    pyc_operators pyc_nusol pyc_graph
    cparticle_template ctypecasting
)

# Check if CUDA is enabled for AnalysisG
if (CMAKE_ANALYSISG_CUDA)
# Enable separate compilation for the cpyc CUDA target
set_property(TARGET cpyc PROPERTY CUDA_SEPARATE_COMPILATION ON)
# Set public compile options for the cpyc target, specifically enabling experimental relaxed constexpr for CUDA
target_compile_options(cpyc PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
# Define a public compile definition PYC_CUDA for the cpyc target, indicating CUDA is used
target_compile_definitions(cpyc PUBLIC PYC_CUDA ON)
# Link the cpyc target with the CUDA driver library (publicly)
target_link_libraries(cpyc PUBLIC CUDA::cuda_driver)
endif()
# Set private compile options for the cpyc target, enabling Position Independent Code (PIC)
target_compile_options(cpyc PRIVATE -fPIC)
# Install the cpyc target (the compiled library) to the default destination directory
install(TARGETS cpyc DESTINATION .)
