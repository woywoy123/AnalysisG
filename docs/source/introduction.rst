Introduction to AnalysisG
==========================

AnalysisG is a high-performance analysis framework designed for High Energy Physics (HEP) applications. The framework combines the performance of C++ and CUDA with the ease of use of Python through Cython bindings.

Key Features
------------

* **Multi-Language Support**: C++, CUDA, and Cython implementations
* **High Performance**: Optimized for large-scale HEP data analysis
* **Graph Neural Networks**: Built-in support for GNN-based analyses
* **Extensible Architecture**: Template-based design for easy customization
* **GPU Acceleration**: CUDA implementations for compute-intensive operations

Architecture Overview
---------------------

The framework is organized into several key packages:

Core Package
~~~~~~~~~~~~
Contains base template classes and core functionality that users extend to implement custom analyses.

Events Package
~~~~~~~~~~~~~~
Implements event-level data structures for various physics analyses (BSM 4-tops, MC20, etc.).

Graphs Package
~~~~~~~~~~~~~~
Provides graph representations of physics events for GNN-based analyses.

Metrics Package
~~~~~~~~~~~~~~~
Implements various metrics for model evaluation (accuracy, PageRank, etc.).

Models Package
~~~~~~~~~~~~~~
Contains machine learning model implementations (GRIFT, Recursive GNN, etc.).

Modules Package
~~~~~~~~~~~~~~~
Low-level C++ implementations of core algorithms and data structures.

PyC Package
~~~~~~~~~~~
Python-C++ interface layer with CUDA-accelerated operations for physics calculations, graph operations, and transformations.

Design Philosophy
-----------------

AnalysisG follows a two-tier architecture:

1. **Simple Interfaces**: High-level template classes that users inherit from and override to implement custom behavior.
2. **Complex Technical Components**: Low-level C++/CUDA implementations that provide high-performance operations.

This design allows users to focus on physics analysis while leveraging highly optimized backend implementations.
