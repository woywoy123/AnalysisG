base.cuh
========

**File Path**: ``src/AnalysisG/pyc/graph/include/graph/base.cuh``

**File Type**: CUDA Header

**Lines**: 81

Description
-----------

const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

Dependencies
------------

**C++ Includes**:

- ``utils/atomic.cuh``

CUDA Kernels
------------

- ``_edge_summing()`` (CUDA kernel)
- ``_fast_unique()`` (CUDA kernel)
- ``_prediction_topology()`` (CUDA kernel)
- ``_unique_sum()`` (CUDA kernel)

