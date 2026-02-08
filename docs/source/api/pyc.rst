PyC Package
===========

PyC (Python CUDA) is a high-performance package for physics calculations with native C++ and CUDA implementations.

Overview
--------

PyC is a self-contained package located in ``pyc/pyc/`` that provides:

- High-performance physics calculations
- Native C++ implementations
- CUDA acceleration for GPU computing
- LibTorch integration
- Tensor operations
- Graph algorithms

PyC can be used independently of the main AnalysisG framework or integrated seamlessly with it.

Key Features
------------

Performance
~~~~~~~~~~~

- **C++ Backend**: All algorithms implemented in optimized C++
- **CUDA Kernels**: Native GPU acceleration for supported operations
- **LibTorch**: Integration with PyTorch's tensor library
- **Vectorization**: SIMD optimizations where applicable

Functionality
~~~~~~~~~~~~~

- Physics calculations (ΔR, invariant mass, etc.)
- Coordinate transformations
- Graph operations
- Tensor operations
- Neutrino reconstruction
- Plotting utilities

Architecture
------------

PyC consists of several submodules:

.. toctree::
   :maxdepth: 2

   pyc/physics
   pyc/operators
   pyc/graph
   pyc/transform
   pyc/nusol
   pyc/tools

Submodule Overview
------------------

Physics
~~~~~~~

Physics calculations including:

- Delta R (ΔR) calculation
- Invariant mass
- Transverse mass
- Angular distances
- Momentum operations

See :doc:`pyc/physics` for details.

Operators
~~~~~~~~~

Tensor operations:

- Element-wise operations
- Aggregations (sum, mean, etc.)
- Graph-based aggregations
- Custom operators

See :doc:`pyc/operators` for details.

Graph
~~~~~

Graph algorithms:

- Edge construction
- Node aggregation
- Graph pooling
- PageRank
- Community detection

See :doc:`pyc/graph` for details.

Transform
~~~~~~~~~

Coordinate transformations:

- Cartesian to polar
- Polar to Cartesian
- Boost transformations
- Rotation matrices

See :doc:`pyc/transform` for details.

NuSol
~~~~~

CUDA-accelerated neutrino reconstruction:

- Batch processing
- Multiple algorithms
- GPU acceleration
- Tensor interface

See :doc:`pyc/nusol` for details.

Tools
~~~~~

Utility functions:

- I/O operations
- String utilities
- Data conversion
- Helper functions

See :doc:`pyc/tools` for details.

Installation
------------

PyC is automatically built with AnalysisG when CUDA is available:

.. code-block:: bash

   # CUDA will be detected automatically
   pip install .

To disable CUDA support:

.. code-block:: bash

   # Set environment variable
   export DISABLE_CUDA=1
   pip install .

Usage Examples
--------------

Standalone Usage
~~~~~~~~~~~~~~~~

PyC can be used without the AnalysisG framework:

.. code-block:: python

   import pyc
   
   # Physics calculations
   delta_r = pyc.physics.DeltaR(eta1, phi1, eta2, phi2)
   mass = pyc.physics.InvariantMass(pt1, eta1, phi1, E1, pt2, eta2, phi2, E2)
   
   # Tensor operations
   result = pyc.operators.aggregate(tensor, indices, operation='sum')

With AnalysisG
~~~~~~~~~~~~~~

PyC integrates seamlessly with AnalysisG:

.. code-block:: python

   from AnalysisG.core.event_template import EventTemplate
   import pyc
   
   class MyEvent(EventTemplate):
       def process(self):
           # Use PyC for calculations
           self.delta_r = pyc.physics.DeltaR(
               self.jet_eta, self.jet_phi,
               self.lepton_eta, self.lepton_phi
           )

GPU Acceleration
----------------

CUDA Support
~~~~~~~~~~~~

When CUDA is available, operations automatically use GPU:

.. code-block:: python

   import torch
   import pyc
   
   # Move tensors to GPU
   tensor_gpu = tensor.cuda()
   
   # Operations run on GPU
   result = pyc.operators.aggregate(tensor_gpu, indices)

Tensor vs CUDA Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~

PyC provides two interfaces:

- **Tensor Interface**: Uses LibTorch tensors (CPU or GPU)
- **CUDA Interface**: Direct CUDA kernel calls (GPU only)

.. code-block:: python

   # Tensor interface (automatically uses GPU if available)
   import pyc
   result = pyc.physics.DeltaR(eta1, phi1, eta2, phi2)
   
   # Direct CUDA interface (requires GPU)
   from pyc.cuda import physics
   result = physics.DeltaR(eta1_gpu, phi1_gpu, eta2_gpu, phi2_gpu)

Performance Benchmarks
----------------------

PyC provides significant speedups over pure Python:

- **Physics calculations**: 10-100x faster than numpy
- **Graph operations**: 5-50x faster than networkx
- **Tensor operations**: Comparable to native PyTorch
- **GPU operations**: 100-1000x faster for large batches

See ``pyc/benchmarks/`` for detailed benchmarks.

C++ API
-------

PyC also provides a C++ API for direct use in C++ code:

.. code-block:: cpp

   #include <pyc/physics.h>
   
   // Calculate delta R
   float dr = Physics::DeltaR(eta1, phi1, eta2, phi2);
   
   // Invariant mass
   float mass = Physics::InvariantMass(pt1, eta1, phi1, E1, 
                                       pt2, eta2, phi2, E2);

Building PyC
------------

PyC is built using CMake:

.. code-block:: bash

   mkdir build
   cd build
   cmake ..
   make -j$(nproc)

For CUDA support:

.. code-block:: bash

   cmake -DUSE_CUDA=ON ..
   make -j$(nproc)

Dependencies
------------

Required:

- C++17 compatible compiler
- CMake >= 3.23
- LibTorch (PyTorch C++ library)

Optional:

- CUDA Toolkit >= 11.0 (for GPU acceleration)
- cuDNN (for advanced GPU features)

See Also
--------

* :doc:`core` - Core AnalysisG API
* :doc:`modules` - Modules API
* `LibTorch Documentation <https://pytorch.org/cppdocs/>`_
