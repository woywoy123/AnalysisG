PyC Package Overview
====================

The PyC package provides Python-C++ interface with CUDA acceleration.

Overview
--------

This package bridges Python and C++/CUDA implementations for high-performance computing.

Components
----------

cutils
~~~~~~

CUDA utility functions and atomic operations.

* **Files**: ``atomic.cu``, ``utils.cu``, ``utils.cxx``
* **Headers**: ``include/utils/atomic.cuh``, ``include/utils/utils.cuh``, ``include/utils/utils.h``

graph
~~~~~

GPU-accelerated graph operations.

* **Files**: ``graph.cu``, ``graph.cxx``, ``pagerank.cu``, ``reconstruction.cu``
* **Headers**: ``include/graph/*.cuh``, ``include/graph/*.h``

interface
~~~~~~~~~

Python-C++ interface layer.

* **Files**: ``graph.cxx``, ``interface.cxx``, ``nusol.cxx``, ``operators.cxx``, ``physics.cxx``, ``transform.cxx``
* **Header**: ``include/pyc/pyc.h``

nusol
~~~~~

Neutrino reconstruction with CPU and GPU implementations.

* **CUDA Files**: ``cuda/*.cu``
* **CPU Files**: ``tensor/*.cxx``
* **Headers**: ``include/nusol/*.cuh``, ``include/nusol/*.h``

operators
~~~~~~~~~

Mathematical operators with GPU acceleration.

* **Files**: ``operators.cu``, ``operators.cxx``
* **Headers**: ``include/operators/*.cuh``, ``include/operators/*.h``

physics
~~~~~~~

Physics calculations with GPU support.

* **Files**: ``physics.cu``, ``physics.cxx``
* **Headers**: ``include/physics/*.cuh``, ``include/physics/*.h``

transform
~~~~~~~~~

Coordinate transformations.

* **Files**: ``transform.cu``, ``transform.cxx``
* **Headers**: ``include/transform/*.cuh``, ``include/transform/*.h``

See Also
--------

* :doc:`../technical/overview`: Technical overview
* :doc:`../technical/cuda_api`: CUDA API reference
