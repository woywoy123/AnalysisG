CUDA API Reference
===================

This section documents the CUDA API for GPU-accelerated operations.

Overview
--------

The CUDA API provides GPU implementations for compute-intensive operations.

CUDA Components
---------------

The following modules have CUDA implementations:

* **pyc/cutils**: Atomic operations and utilities
* **pyc/graph**: Graph operations and PageRank
* **pyc/nusol**: Neutrino reconstruction
* **pyc/operators**: Mathematical operators
* **pyc/physics**: Physics calculations
* **pyc/transform**: Coordinate transformations

CUDA Features
-------------

* Kernel implementations for parallel processing
* Device memory management
* Automatic fallback to CPU when CUDA unavailable
* Optimized for modern GPUs (Compute Capability 6.0+)

See Also
--------

* :doc:`overview`: Technical overview
* :doc:`cpp_api`: C++ API reference
* :doc:`../pyc/overview`: PyC package overview
