# Check if CUDA is enabled for AnalysisG
if(CMAKE_ANALYSISG_CUDA)
    # Define the source files for CUDA compilation
    set(SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/atomic.cuh # CUDA header for atomic operations
        ${CMAKE_CURRENT_SOURCE_DIR}/atomic.cu # CUDA source for atomic operations

        ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/utils.cuh # CUDA header for utility functions
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.cu # CUDA source for utility functions
    )
    # Add a library target named pyc_utils using the defined source files
    add_library(pyc_utils ${SOURCE_FILES})
    # Set compile options for the pyc_utils target, specifically enabling experimental relaxed constexpr for CUDA
    target_compile_options(pyc_utils PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
    # Enable separate compilation for the pyc_utils CUDA target
    set_property(TARGET pyc_utils PROPERTY CUDA_SEPARATE_COMPILATION ON)
else() # If CUDA is not enabled
    # Define the source files for C++ compilation
    set(SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/utils.h # C++ header for utility functions
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.cxx # C++ source for utility functions
    )
    # Add a library target named pyc_utils using the defined source files
    add_library(pyc_utils ${SOURCE_FILES})
endif() # End of the CUDA check

# Add include directories for the pyc_utils target (private)
target_include_directories(pyc_utils PRIVATE include/cutils)
# Add include directories for the pyc_utils target (public)
target_include_directories(pyc_utils PUBLIC include)
# Link the pyc_utils target with Torch libraries
target_link_libraries(pyc_utils PUBLIC ${TORCH_LIBRARIES})
# Set compile options for the pyc_utils target, enabling Position Independent Code (PIC)
target_compile_options(pyc_utils PRIVATE -fPIC)
