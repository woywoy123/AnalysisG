pagerank.cu
===========

**File Path**: ``src/AnalysisG/pyc/graph/pagerank.cu``

**File Type**: CUDA Source

**Lines**: 259

Description
-----------

const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

Dependencies
------------

**C++ Includes**:

- ``graph/pagerank.cuh``
- ``utils/atomic.cuh``
- ``utils/utils.cuh``

CUDA Kernels
------------

- ``_get_max_node()`` (CUDA kernel)
- ``_get_remapping()`` (CUDA kernel)
- ``_page_rank()`` (CUDA kernel)

