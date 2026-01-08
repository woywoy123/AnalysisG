.. _installation:

====================
Installation
====================

System Requirements
------------------
Before installing AnalysisG, ensure the following prerequisites are met:

*   **C++ Compiler**: GCC 7+ (9.3.0+ recommended) or Clang 5+ (10.0.0+ recommended) with C++17 support.
*   **CMake**: Version 3.15 or higher.
*   **Python**: Version 3.8 or higher.
*   **ROOT**: A working ROOT installation (e.g., `root-framework` via snap or from https://root.cern/). Ensure ROOT environment variables are set (e.g., `source /path/to/root/bin/thisroot.sh`).
*   **HDF5**: Development libraries (e.g., `libhdf5-dev` on Debian/Ubuntu).
*   **RapidJSON**: Automatically downloaded and built by CMake via `FetchContent`.
*   **LibTorch (PyTorch C++ API)**: Automatically downloaded and built by CMake via `FetchContent`.

PyTorch Compatibility
---------------------

.. warning::
    AnalysisG requires a specific ABI-compatible version of PyTorch C++. If your Python script uses both the `torch` Python package and the AnalysisG framework, you **must** uninstall any existing `torch` package and install the correct ABI-compatible version listed below. Failure to do so will likely result in `ImportError: ... undefined symbol: _ZN...` errors due to ABI mismatches between the LibTorch used by AnalysisG and the `torch` Python package. See PyTorch issue #51039 for background.

The framework is currently compiled against **LibTorch 2.7.0**. To ensure compatibility, install the corresponding Python package:

.. code-block:: bash

    # For CPU-only support:
    pip uninstall torch -y
    pip install "torch==2.7.0+cpu" --index-url https://download.pytorch.org/whl/cpu

    # For CUDA 12.1 support (ensure your system CUDA matches):
    pip uninstall torch -y
    pip install "torch==2.7.0+cu126" --index-url https://download.pytorch.org/whl/cu126

    # Check PyTorch download page for other CUDA versions if needed:
    # https://download.pytorch.org/whl/torch_stable.html

Installation from Source Code
-----------------------------

For the latest development version or if contributing:

1.  **Ensure Prerequisites**: Verify all system requirements, especially the correct PyTorch version, are met.
2.  **Clone the repository**:
     .. code-block:: bash

         git clone https://github.com/woywoy123/AnalysisG.git
         cd AnalysisG

3.  **Create build directory**:
     .. code-block:: bash

         mkdir build && cd build

4.  **Configure, Compile, and Install**:
     .. code-block:: bash

         # Configure using CMake (downloads LibTorch/RapidJSON)
         cmake ..

         # Compile (adjust -j based on your CPU cores)
         make -j$(nproc)

         # Second cmake call installs the package to Python's site-packages
         cmake ..

     *Note*: The compilation process, especially for CUDA kernels, can be computationally intensive and time-consuming. 
             The second `cmake ..` command locates the Python `site-packages` directory and copies the built library there, making it importable.

Troubleshooting
--------------

This section covers common issues encountered during installation and runtime.

Installation Problems
~~~~~~~~~~~~~~~~~~~~~

**Problem: PyTorch ABI Incompatibility / ImportError**

*   **Symptom**: Errors like `ImportError: ... undefined symbol: _ZN...` when importing `analysisg` in Python.
*   **Cause**: Mismatch between the LibTorch C++ library used by AnalysisG and the `torch` Python package installed in your environment.
*   **Solution**: Ensure you have uninstalled other PyTorch versions and installed the specific ABI-compatible version mentioned in the "PyTorch Compatibility" section above. Verify the installation:
     .. code-block:: bash

          python -c "import torch; print(torch.__version__)"
          python -c "import analysisg; print(analysisg.__version__)"

**Problem: CMake cannot find ROOT**

*   **Symptom**: An error message appears during the `cmake ..` step stating that CMake cannot find ROOT.
*   **Solution**:
     1.  Confirm ROOT is installed correctly: `root-config --version`.
     2.  Ensure ROOT environment variables are sourced: `source /path/to/root/bin/thisroot.sh`. Check `echo $ROOTSYS`.
     3.  If needed, explicitly tell CMake where ROOT is: `cmake -DROOT_DIR=/path/to/root ..`

**Problem: CMake cannot find Boost libraries**

*   **Symptom**: CMake reports that it cannot find the Boost libraries during the `cmake ..` step.
*   **Solution**:
     1.  Install the required Boost development packages (e.g., `sudo apt-get install libboost-all-dev` or `brew install boost`).
     2.  If needed, explicitly tell CMake where Boost is: `cmake -DBOOST_ROOT=/path/to/boost ..`

**Problem: Compilation errors during `make`**

*   **Symptom**: The `make` command fails with compiler errors.
*   **Solution**:
     1.  Verify your C++ compiler meets the minimum version requirements (GCC 7+/Clang 5+, newer recommended). Check with `g++ --version` or `clang --version`.
     2.  Ensure all prerequisites (ROOT, HDF5, Boost) are installed correctly.
     3.  Check CMake output for errors during dependency fetching (LibTorch, RapidJSON).
     4.  Clean the build directory and try again:
          .. code-block:: bash

              cd build
              make clean
              rm -rf *  # Or just rm CMakeCache.txt
              cmake ..
              make -j$(nproc)

**Problem: CUDA Version Mismatches**

*   **Symptom**: Errors related to CUDA during compilation or runtime when using the GPU-enabled version.
*   **Solution**: Ensure your system's installed CUDA toolkit version (check with `nvcc --version`) is compatible with the CUDA version specified in the PyTorch package name (e.g., `+cu121` requires CUDA 12.1 compatible drivers and toolkit).

Runtime Problems
~~~~~~~~~~~~~~~~

**Problem: Segmentation Fault during execution**

*   **Symptom**: The program crashes unexpectedly with a "Segmentation Fault".
*   **Solution**:
     1.  Run the program using a debugger like GDB: `gdb --args ./your_executable [arguments]`.
     2.  Inside GDB, type `run` to start the program. When it crashes, type `bt` (backtrace) to see the call stack and identify where the crash occurred.
     3.  Check your input files for corruption or incorrect formatting.
     4.  Review your custom C++ or Python code for potential issues like accessing null pointers or out-of-bounds array access.

**Problem: Type errors in Python API**

*   **Symptom**: `TypeError` or `AttributeError` when calling AnalysisG functions from Python.
*   **Solution**:
     1.  Consult the API documentation for the expected data types of function arguments.
     2.  Ensure NumPy arrays passed to AnalysisG have the correct `dtype` (e.g., `np.float32`, `np.int64`).
          .. code-block:: python

                import numpy as np
                correct_array = np.array([1.0, 2.0], dtype=np.float32)
     3.  Use provided helper functions if available for creating complex objects.

Metadata Problems
~~~~~~~~~~~~~~~~~

**Problem: Missing or invalid metadata**

*   **Symptom**: Errors related to missing or invalid metadata fields when reading or processing files.
*   **Solution**:
     1.  Inspect the metadata of your input files:
          .. code-block:: python

                import analysisg as ag
                dataset = ag.Dataset("your_data.root")
                print(dataset.get_metadata())
     2.  Add or correct metadata fields as needed using the appropriate API functions before saving or processing.
          .. code-block:: python

                dataset.set_metadata("some_key", "some_value")
                dataset.save("output_data.root") # If applicable

Data Processing Problems
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem: Slow data processing**

*   **Symptom**: Analysis jobs take an unexpectedly long time to complete.
*   **Solution**:
     1.  **Multithreading**: Enable multithreading if your analysis is parallelizable.
          .. code-block:: cpp
                // C++
                analysis.set_num_threads(std::thread::hardware_concurrency());

          .. code-block:: python
                # Python
                import os
                analysis.set_num_threads(os.cpu_count())
     2.  **Caching**: Enable caching if the same data is accessed repeatedly (consult documentation for applicability).
     3.  **Filtering**: Apply event or object selections (filters) as early as possible in your analysis chain to reduce the amount of data processed in later stages.
     4.  **Optimize Code**: Review custom C++ or Python functions (extractors, processors) for performance bottlenecks.

**Problem: Errors in user-defined functions**

*   **Symptom**: Crashes or incorrect results originating from custom code (e.g., C++ lambdas, Python callables) passed to the framework.
*   **Solution**:
     1.  **Defensive Coding**: Add checks for potential issues like empty containers or null pointers before accessing elements.
          .. code-block:: cpp
                // Example: Safely access leading jet pT
                auto extractor = [](const event_t& evt) {
                      if (evt.jets.empty()) return 0.0; // Handle empty case
                      return evt.jets[0].pt;
                };
     2.  **Logging/Debugging**: Increase log verbosity or add print statements within your custom functions to trace execution and variable values.
          .. code-block:: cpp
                analysis.set_log_level(LogLevel::DEBUG); // C++ example
     3.  **Isolate**: Test the problematic function in isolation with sample inputs to pinpoint the error.

Visualization Problems
~~~~~~~~~~~~~~~~~~~~~~

**Problem: Errors when creating plots or empty plots**

*   **Symptom**: Plotting functions fail, or generated plots are empty.
*   **Solution**:
     1.  **Check Data**: Verify that the histogram or data structure you are trying to plot actually contains data.
          .. code-block:: cpp
                // C++ ROOT Histogram example
                auto h = analysis.get_histogram("my_hist");
                if (h && h->GetEntries() > 0) {
                     // Proceed with plotting
                } else {
                     std::cerr << "Histogram 'my_hist' is empty or invalid!" << std::endl;
                }
          .. code-block:: python
                # Python (assuming a plotting function returns a plot object)
                plot_object = plot_histogram(data)
                if plot_object is None: # Or check data source directly
                      print("Failed to generate plot, check data.")

     2.  **Dependencies**: Ensure plotting libraries (e.g., Matplotlib in Python) are installed correctly.

**Problem: Incorrect histogram scaling or normalization**

*   **Symptom**: Plotted histograms do not reflect the expected scale, normalization, or weighting.
*   **Solution**:
     1.  **Weighting**: Confirm if event weights should be used and if they are enabled correctly in the analysis configuration.
     2.  **Normalization**: Check the options of your plotting function. Many libraries offer flags to normalize to unity, area, or a specific cross-section/luminosity.
          .. code-block:: python
                # Example using a hypothetical plotting function
                plot_histogram(h, normalize='unit_area', scale_factor=1000)
     3.  **Log Scale**: For distributions spanning several orders of magnitude, consider using a logarithmic y-axis for better visualization.
          .. code-block:: python
                import matplotlib.pyplot as plt
                plot_histogram(h) # Assuming this plots on the current axes
                plt.yscale('log')
                plt.show()

