Installation
============

Requirements
------------

* Python 3.8 or later
* C++17 compatible compiler (GCC 9+, Clang 10+, or MSVC 2019+)
* CUDA Toolkit 11.0 or later (optional, for GPU acceleration)
* CMake 3.18 or later
* Cython 0.29 or later

Dependencies
------------

The framework requires the following Python packages:

* numpy
* torch (PyTorch)
* cython

Building from Source
--------------------

.. warning::
   Do not attempt to compile the main package directly. Use the provided build scripts.

.. code-block:: bash

   git clone https://github.com/woywoy123/AnalysisG.git
   cd AnalysisG
   pip install -e .

CUDA Support
------------

To build with CUDA support, ensure that:

1. CUDA Toolkit is installed and available in your PATH
2. The ``nvcc`` compiler is accessible
3. CMake can detect your CUDA installation

The build system will automatically detect CUDA and enable GPU-accelerated operations if available.

Verifying Installation
----------------------

To verify that AnalysisG is installed correctly:

.. code-block:: python

   import AnalysisG
   print(AnalysisG.__version__)

Troubleshooting
---------------

If you encounter build errors:

1. Ensure all dependencies are installed
2. Check that your compiler supports C++17
3. Verify CUDA installation (if using GPU features)
4. Check CMake configuration output for errors
