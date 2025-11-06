Graph Component
===============

Python-C++ interface for graph operations.

File Location
~~~~~~~~~~~~~

* **Base Path**: ``src/AnalysisG/pyc/graph/``
* **C++ Files**: ``src/AnalysisG/pyc/graph/*.cxx``
* **CUDA Files**: GPU-accelerated implementations
* **Headers**: ``src/AnalysisG/pyc/graph/include/**/*.h``

Description
-----------

This component provides high-performance graph operations with optional GPU acceleration.

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
