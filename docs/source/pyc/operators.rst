Operators Component
===================

Python-C++ interface for operators operations.

File Location
~~~~~~~~~~~~~

* **Base Path**: ``src/AnalysisG/pyc/operators/``
* **C++ Files**: ``src/AnalysisG/pyc/operators/*.cxx``
* **CUDA Files**: GPU-accelerated implementations
* **Headers**: ``src/AnalysisG/pyc/operators/include/**/*.h``

Description
-----------

This component provides high-performance operators operations with optional GPU acceleration.

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
