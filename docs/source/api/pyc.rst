PyC - High Performance Computing
=================================

The ``pyc`` (Python CUDA) module provides high-performance C++ and CUDA implementations integrated with PyTorch.

Overview
--------

PyC is a standalone package within AnalysisG that implements computationally intensive algorithms in native C++ and CUDA. It uses the LibTorch API for seamless integration with PyTorch workflows.

Key features:

- **Native C++/CUDA**: Performance-critical code written in C++ with optional CUDA acceleration
- **PyTorch Integration**: Uses LibTorch for tensor operations and GPU memory management
- **Zero Copy**: Efficient memory sharing between Python and C++
- **Thread Safe**: Designed for multi-threaded Python environments

Submodules
----------

The pyc module is organized into several specialized submodules:

CUtils
~~~~~~

Core utility functions and data structures.

Graph Operations
~~~~~~~~~~~~~~~~

Graph-level operations for neural network processing:

- Node aggregation
- Edge operations
- Graph pooling
- Message passing primitives

Interface
~~~~~~~~~

Python/C++ interface layer providing Cython bindings.

NuSol - Neutrino Solver
~~~~~~~~~~~~~~~~~~~~~~~

Analytical and numerical neutrino momentum reconstruction:

- Single neutrino reconstruction
- Double neutrino reconstruction
- CUDA-accelerated solvers
- Tensor-based batch processing

Operators
~~~~~~~~~

Mathematical operators optimized for physics:

- DeltaR calculations
- Invariant mass computation
- Angular separations
- Momentum operations

Physics
~~~~~~~

Physics-specific calculations:

- Coordinate transformations (polar ↔ cartesian)
- Lorentz transformations
- 4-vector operations
- Kinematic calculations

Transform
~~~~~~~~~

Data transformation utilities:

- Batch transformations
- Feature scaling
- Coordinate system conversions

Architecture
------------

PyC follows a layered architecture:

1. **CUDA Kernels** (lowest level): Raw CUDA code for GPU operations
2. **C++ Wrappers**: C++ classes wrapping CUDA kernels with LibTorch tensors
3. **Cython Interface**: Python-accessible interface via Cython
4. **Python API**: High-level Python functions

This design allows:

- Using PyC from pure Python (via Cython)
- Using PyC from Cython code (direct C++ access)
- Using PyC from pure C++ projects (standalone)

Performance
-----------

PyC is designed for maximum performance:

- **GPU Acceleration**: CUDA kernels for parallel operations
- **Memory Efficiency**: Minimal memory allocation and copying
- **Batch Processing**: Vectorized operations across multiple events
- **Multi-threading**: Thread-safe for concurrent Python execution

Example Use Cases
-----------------

PyC is used throughout AnalysisG for:

1. **Data Preprocessing**: Fast coordinate transformations and feature engineering
2. **Graph Construction**: Efficient edge and node feature computation
3. **Neutrino Reconstruction**: Solving complex kinematics constraints
4. **Training**: Custom loss functions and metrics
5. **Inference**: High-throughput model evaluation
