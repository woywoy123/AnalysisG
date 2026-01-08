============
Installation
============

This guide covers the installation of AnalysisG and its dependencies.

Prerequisites
=============

Before installing AnalysisG, ensure you have the following:

System Requirements
-------------------

- **Operating System**: Linux (Ubuntu 20.04+, CentOS 7+, etc.)
- **Compiler**: GCC 8+ or Clang 10+ with C++17 support
- **CMake**: Version 3.18 or later
- **Python**: 3.8 or later

Required Dependencies
---------------------

- **PyTorch (LibTorch)**: Version 1.10 or later
- **ROOT** (optional): For reading ROOT files
- **HDF5**: For data storage
- **Cython**: For Python bindings

Installation Methods
====================

From Source
-----------

Clone the repository and build:

.. code-block:: bash

    git clone https://github.com/woywoy123/AnalysisG.git
    cd AnalysisG
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    make install

Using pip
---------

Install the Python package:

.. code-block:: bash

    pip install analysisg

Conda Environment
-----------------

Create a conda environment with all dependencies:

.. code-block:: bash

    conda create -n analysisg python=3.10
    conda activate analysisg
    pip install analysisg

Configuration
=============

CMake Options
-------------

The following CMake options are available:

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``BUILD_PYTHON``
     - ON
     - Build Python bindings
   * - ``USE_CUDA``
     - OFF
     - Enable CUDA support
   * - ``USE_ROOT``
     - ON
     - Enable ROOT file support

Verification
============

To verify your installation:

.. code-block:: python

    import AnalysisG
    print(AnalysisG.__version__)

If no errors occur, the installation was successful.
