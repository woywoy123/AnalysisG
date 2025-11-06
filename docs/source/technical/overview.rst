Complex Technical Components
==============================

The technical components of AnalysisG consist of high-performance C++/CUDA implementations that provide the computational backbone for the framework.

Overview
--------

These components are designed for:

* Maximum performance through C++17 optimizations
* GPU acceleration via CUDA
* Low-level memory management
* Efficient algorithms for HEP data processing

When to Use Technical Components
---------------------------------

Technical components are used internally by the simple interfaces. Most users will interact with these through the high-level Python/Cython APIs. Direct use of technical components is recommended only when:

* Implementing new low-level algorithms
* Optimizing performance-critical code
* Adding GPU-accelerated operations
* Extending the framework with new functionality

Component Categories
--------------------

Modules Package
~~~~~~~~~~~~~~~

The modules package contains pure C++ implementations of core algorithms and data structures. These are complex technical components with private member variables and advanced C++ features.

**Key modules:**

* **analysis**: Analysis framework implementation
* **container**: Data container implementations
* **dataloader**: Data loading and preprocessing
* **event**: Event-level C++ implementations
* **graph**: Graph data structures and algorithms
* **io**: Input/output operations
* **meta**: Metadata management
* **metric**: Metric computation engines
* **model**: Model backend implementations
* **nusol**: Neutrino reconstruction algorithms
* **optimizer**: Optimization algorithms
* **particle**: Particle data structures
* **selection**: Selection logic implementations
* **structs**: Core data structures
* **tools**: Utility functions

PyC Package
~~~~~~~~~~~

The PyC package provides Python-C++ interface layers with CUDA acceleration for computationally intensive operations.

**Key components:**

* **cutils**: CUDA utility functions and atomic operations
* **graph**: GPU-accelerated graph operations and PageRank
* **interface**: Python-C++ interface layer
* **nusol**: Neutrino reconstruction (CPU and GPU)
* **operators**: Mathematical operators (CPU and GPU)
* **physics**: Physics calculations (CPU and GPU)
* **transform**: Coordinate transformations (CPU and GPU)

Architecture
------------

The technical components follow a layered architecture:

1. **C++ Core Layer**: Pure C++ implementations in ``modules/``
2. **CUDA Acceleration Layer**: GPU implementations in ``pyc/*/cuda/`` and ``*.cu`` files
3. **Interface Layer**: Cython bindings in ``pyc/interface/``
4. **Python API Layer**: High-level Python APIs in ``core/``

File Organization
-----------------

Technical components follow a consistent structure:

.. code-block:: text

   module_name/
   ├── CMakeLists.txt          # Build configuration
   ├── cxx/                    # C++ implementation files
   │   ├── file1.cxx
   │   └── file2.cxx
   ├── cuda/                   # CUDA implementations (if applicable)
   │   ├── kernel1.cu
   │   └── kernel2.cu
   └── include/                # Header files
       └── module_name/
           ├── header1.h
           └── header2.h

See Also
--------

* :doc:`modules`: Modules package documentation
* :doc:`cpp_api`: C++ API reference
* :doc:`cuda_api`: CUDA API reference
