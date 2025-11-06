Transform Component
===================

Python-C++ interface for transform operations.

File Location
~~~~~~~~~~~~~

* **Base Path**: ``src/AnalysisG/pyc/transform/``
* **C++ Files**: ``src/AnalysisG/pyc/transform/*.cxx``
* **CUDA Files**: GPU-accelerated implementations
* **Headers**: ``src/AnalysisG/pyc/transform/include/**/*.h``

Description
-----------

This component provides high-performance transform operations with optional GPU acceleration.

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
