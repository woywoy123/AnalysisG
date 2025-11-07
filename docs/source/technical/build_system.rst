Build System and CMake Configuration
====================================

This document provides comprehensive documentation for building AnalysisG from source, including CMake configuration, dependencies, and build options.

Overview
--------

AnalysisG uses **CMake 3.20+** as its build system with support for:

* C++20 standard with modern features
* CUDA integration (optional) for GPU-accelerated computations
* PyTorch LibTorch backend for ML operations
* ROOT framework for HEP data I/O
* HDF5 for efficient data storage
* RapidJSON for JSON parsing

Build Architecture
------------------

The project is organized as a **hybrid Python/C++ package** with:

* **Python Frontend**: User-facing API in ``src/AnalysisG/``
* **C++ Backend**: High-performance modules in ``src/AnalysisG/modules/``
* **CUDA Extensions**: GPU kernels in ``pyc/`` (optional)

**Build Flow**::

    1. CMake configures dependencies (LibTorch, ROOT, HDF5)
    2. C++ modules compiled → installed to site-packages
    3. CUDA extensions compiled (if enabled) → installed to pyc/
    4. Python wrappers link to compiled libraries

Root CMakeLists.txt
-------------------

**Location**: ``/CMakeLists.txt``

Project Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
   
   set(SKBUILD_PROJECT_NAME AnalysisG)
   project(${SKBUILD_PROJECT_NAME} VERSION 5.0 LANGUAGES CXX)
   
   set(CMAKE_CXX_STANDARD 20)
   set(CMAKE_CXX_STANDARD_REQUIRED ON)
   set(CMAKE_CUDA_ARCHITECTURES "all")

**Explanation**:

* Requires CMake 3.20 for modern CMake features (FetchContent improvements)
* Project name set to ``AnalysisG`` with semantic versioning
* C++20 standard enabled for concepts, ranges, modules
* CUDA architectures set to "all" for maximum GPU compatibility

Build Flags
~~~~~~~~~~~

.. code-block:: cmake

   set(CMAKE_CXX_FLAGS_DEBUG "-g")
   # set(CMAKE_CXX_FLAGS_RELEASE "-O3")  # Commented - uses default optimizations

**Debug Mode**: Includes symbols for GDB/LLDB debugging

**Release Mode**: Uses default CMake optimization (-O2 typically)

Build Options
~~~~~~~~~~~~~

.. code-block:: cmake

   option(BUILD_DOC "Build documentation" OFF)
   find_package(Doxygen)
   
   set(CMAKE_ANALYSISG_CUDA OFF)

**BUILD_DOC**: Generate C++ API docs with Doxygen (disabled by default)

**CMAKE_ANALYSISG_CUDA**: Enable CUDA extensions (OFF by default, override via environment)

Dependencies
------------

RapidJSON (JSON Parser)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

   set(RAPIDJSON_BUILD_THIRDPARTY_GTEST OFF)
   set(RAPIDJSON_BUILD_EXAMPLES OFF)
   set(RAPIDJSON_BUILD_TESTS OFF)
   set(RAPIDJSON_BUILD_DOC OFF)
   set(RAPIDJSON_BUILD_CXX17 ON)
   
   FetchContent_Declare(rapidjson
       GIT_REPOSITORY "https://github.com/Tencent/rapidjson.git"
       GIT_TAG origin/master
       FIND_PACKAGE_ARGS)
   FetchContent_MakeAvailable(rapidjson)

**Purpose**: Parse JSON configuration files and metadata

**Installation**: Automatically fetched and built by CMake

**Headers**: ``${CMAKE_BINARY_DIR}/_deps/rapidjson-src/include``

PyTorch LibTorch
~~~~~~~~~~~~~~~~

.. code-block:: cmake

   if(CMAKE_ANALYSISG_CUDA)
       FetchContent_Declare(torch 
           URL "https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu126.zip")
   else()
       FetchContent_Declare(torch 
           URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip")
   endif()
   FetchContent_MakeAvailable(torch)
   
   find_package(Torch REQUIRED CONFIG)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

**Purpose**: 

* Tensor operations for ML
* Automatic differentiation
* Model training infrastructure
* CUDA memory management

**Version**: 2.7.0 (CPU or CUDA 12.6)

**Key Variables**:

* ``${TORCH_LIBRARIES}`` - LibTorch libraries to link
* ``${TORCH_CXX_FLAGS}`` - Required compiler flags
* ``${CAFFE2_USE_CUDNN}`` - cuDNN availability status

ROOT Framework
~~~~~~~~~~~~~~

.. code-block:: cmake

   find_package(ROOT COMPONENTS RIO Tree Core REQUIRED)
   include_directories(${ROOT_INCLUDE_DIRS})

**Purpose**: 

* Read/write ROOT files (.root)
* TTree data structures
* Branch/leaf navigation

**Required Components**:

* **RIO**: File I/O
* **Tree**: TTree functionality  
* **Core**: Base classes

**Variables**:

* ``${ROOT_LIBRARIES}`` - Libraries to link
* ``${ROOT_INCLUDE_DIRS}`` - Header directories

HDF5 Library
~~~~~~~~~~~~

.. code-block:: cmake

   find_package(HDF5 REQUIRED COMPONENTS CXX)

**Purpose**: 

* Fast data storage (10x faster than ROOT for large datasets)
* Efficient batch loading for training
* Multi-dimensional array storage

**Component**: CXX (C++ bindings)

**Variables**:

* ``${HDF5_CXX_INCLUDE_DIRS}`` - Include paths
* ``${HDF5_LIBRARIES}`` - Libraries to link

Python Development
~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

   find_package(Python COMPONENTS Interpreter Development REQUIRED)
   
   execute_process(
       COMMAND python -c "import sysconfig; print(sysconfig.get_paths()['platlib'])" 
       OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

**Purpose**:

* Python.h headers for extension modules
* Site-packages installation path

**Variables**:

* ``${Python_INCLUDE_DIRS}`` - Python headers
* ``${PYTHON_SITE_PACKAGES}`` - Installation target

Installation
------------

.. code-block:: cmake

   add_subdirectory(src/AnalysisG)
   file(INSTALL ${CMAKE_BINARY_DIR}/src/AnalysisG 
        DESTINATION ${PYTHON_SITE_PACKAGES})

**Process**:

1. Build all C++ modules in ``src/AnalysisG/``
2. Generate ``.so`` shared libraries
3. Install to Python site-packages directory

CUDA Build (Optional)
---------------------

**Location**: ``pyc/CMakeLists.txt``

Configuration
~~~~~~~~~~~~~

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
   project(pyc VERSION 5.0 LANGUAGES CXX CUDA)
   
   check_language(CUDA)
   enable_language(CUDA)
   
   set(CMAKE_CUDA_ARCHITECTURES "all")

**Purpose**: Build CUDA-accelerated extensions for:

* Physics calculations (DeltaR, mass, momentum)
* Graph operations (PageRank)
* Neutrino reconstruction
* Coordinate transforms

Dependencies
~~~~~~~~~~~~

.. code-block:: cmake

   FetchContent_Declare(torch 
       URL "https://download.pytorch.org/libtorch/cu126/...zip")
   FetchContent_MakeAvailable(torch)
   
   find_package(Torch REQUIRED)
   find_package(pybind11 REQUIRED)
   find_package(Python COMPONENTS Interpreter Development REQUIRED)

**pybind11**: Python-C++ bindings for extension modules

**CUDA 12.6**: Required for libtorch CUDA version

Compilation Flags
~~~~~~~~~~~~~~~~~

.. code-block:: cmake

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
   set(TORCH_USE_CUDA_DSA ON)

**TORCH_USE_CUDA_DSA**: Device-Side Assertions for debugging

Build Instructions
------------------

Standard Build (CPU Only)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install dependencies (Ubuntu/Debian)
   sudo apt-get install cmake build-essential python3-dev
   sudo apt-get install libhdf5-dev libroot-dev
   
   # Clone repository
   git clone https://github.com/woywoy123/AnalysisG.git
   cd AnalysisG
   
   # Build and install
   pip install -e .

**Process**:

1. scikit-build-core invokes CMake
2. Downloads LibTorch (CPU version)
3. Compiles C++ modules
4. Installs to site-packages

CUDA Build (GPU Acceleration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Additional dependencies
   sudo apt-get install nvidia-cuda-toolkit
   
   # Set CUDA flag
   export CMAKE_ANALYSISG_CUDA=ON
   
   # Build with CUDA support
   pip install -e .

**Requirements**:

* NVIDIA GPU with compute capability 5.0+
* CUDA Toolkit 12.6+
* cuDNN (included in LibTorch download)

**Process**:

1. CMake detects CUDA compiler
2. Downloads LibTorch (CUDA 12.6 version)
3. Compiles C++ modules + CUDA extensions
4. Installs both to site-packages

Development Build
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create build directory
   mkdir build && cd build
   
   # Configure with CMake
   cmake .. -DCMAKE_BUILD_TYPE=Debug
   
   # Build
   make -j$(nproc)
   
   # Run tests
   ctest --output-on-failure

**Debug Build**: Includes symbols, disables optimizations

**Parallel Build**: ``-j$(nproc)`` uses all CPU cores

Build Targets
-------------

**Main Targets**:

* ``AnalysisG`` - Full package with all modules
* ``pyc`` - CUDA extensions (if enabled)
* ``doc_doxygen`` - API documentation (if BUILD_DOC=ON)

**Per-Module Targets** (in src/AnalysisG/):

* ``core`` - Core Cython templates
* ``events`` - Event implementations
* ``graphs`` - Graph implementations  
* ``metrics`` - Metric implementations
* ``models`` - Model implementations
* ``modules`` - C++ backend libraries

Module CMakeLists
-----------------

Each module has its own CMakeLists.txt for compilation:

**Pattern**::

    src/AnalysisG/MODULE_NAME/CMakeLists.txt

**Common Structure**:

.. code-block:: cmake

   # Find dependencies
   find_package(ROOT REQUIRED)
   
   # Create library
   add_library(module_name SHARED
       cxx/implementation.cxx
       cxx/helper.cxx)
   
   # Link dependencies
   target_link_libraries(module_name
       ${ROOT_LIBRARIES}
       ${HDF5_LIBRARIES}
       ${TORCH_LIBRARIES})
   
   # Set properties
   set_target_properties(module_name PROPERTIES
       CXX_STANDARD 20
       POSITION_INDEPENDENT_CODE ON)

**Key Points**:

* Each module is a shared library (``.so``)
* Links against ROOT, HDF5, LibTorch as needed
* Position-independent code required for Python extensions

Troubleshooting
---------------

CMake Version Issues
~~~~~~~~~~~~~~~~~~~~

**Error**: ``CMake 3.20 or higher is required``

**Solution**:

.. code-block:: bash

   # Install newer CMake
   pip install cmake --upgrade

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

**Error**: ``Could not find ROOT``

**Solution**:

.. code-block:: bash

   # Install ROOT (Ubuntu)
   sudo apt-get install libroot-dev
   
   # Or set ROOT_DIR manually
   cmake .. -DROOT_DIR=/path/to/root

CUDA Not Found
~~~~~~~~~~~~~~

**Error**: ``CUDA compiler not found``

**Solution**:

.. code-block:: bash

   # Install CUDA toolkit
   sudo apt-get install nvidia-cuda-toolkit
   
   # Or specify CUDA path
   cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

LibTorch ABI Issues
~~~~~~~~~~~~~~~~~~~

**Error**: ``undefined symbol: _ZN2at...``

**Cause**: ABI mismatch between LibTorch and system compiler

**Solution**: Use cxx11 ABI version of LibTorch (default in CMakeLists.txt)

Out of Memory
~~~~~~~~~~~~~

**Error**: ``g++: fatal error: Killed signal terminated program``

**Cause**: Insufficient RAM during compilation

**Solution**:

.. code-block:: bash

   # Build with fewer parallel jobs
   make -j2  # Instead of -j$(nproc)
   
   # Or increase swap space
   sudo fallocate -l 8G /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile

Build Configuration Summary
---------------------------

After configuration, CMake prints:

.. code-block:: text

   -- ROOT: /usr/lib/x86_64-linux-gnu/libCore.so;...
   -- HDF5: /usr/include/hdf5/serial
   -- RAPIDJSON: /build/_deps/rapidjson-src/include
   -- TORCH: /build/_deps/torch-src/lib/libtorch.so;...
   -- CAFFE2_USE_CUDNN is: 1  (or 0 for CPU-only)

**Verify** these paths are correct before building.

See Also
--------

* :doc:`cpp_complete_reference` - C++ module API reference
* :doc:`cuda_actual_api` - CUDA kernel documentation
* :doc:`../core/analysis` - Analysis workflow
* :doc:`../introduction` - Getting started guide
