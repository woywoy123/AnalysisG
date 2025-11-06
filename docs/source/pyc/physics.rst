Physics Component
=================

Python-C++ interface for physics operations.

File Location
~~~~~~~~~~~~~

* **Base Path**: ``src/AnalysisG/pyc/physics/``
* **C++ Files**: ``src/AnalysisG/pyc/physics/*.cxx``
* **CUDA Files**: GPU-accelerated implementations
* **Headers**: ``src/AnalysisG/pyc/physics/include/**/*.h``

Description
-----------

This component provides high-performance physics operations with optional GPU acceleration.

Features
--------

* CPU implementation in C++
* GPU implementation in CUDA
* Python interface through Cython
* Automatic device selection (CPU/GPU)

See Also
--------

* :doc:`overview`: PyC package overview
* :doc:`../technical/cuda_api`: CUDA API reference
