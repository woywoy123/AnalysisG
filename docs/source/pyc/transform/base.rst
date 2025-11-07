base.cuh
========

**File Path**: ``src/AnalysisG/pyc/transform/include/transform/base.cuh``

**File Type**: CUDA Header

**Lines**: 163

Description
-----------

const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

Dependencies
------------

**C++ Includes**:

- ``utils/atomic.cuh``

CUDA Kernels
------------

- ``EtaK()`` (CUDA kernel)
- ``PhiK()`` (CUDA kernel)
- ``PtEtaPhiEK()`` (CUDA kernel)
- ``PtEtaPhiK()`` (CUDA kernel)
- ``PtK()`` (CUDA kernel)
- ``PxK()`` (CUDA kernel)
- ``PxPyPzEK()`` (CUDA kernel)
- ``PxPyPzK()`` (CUDA kernel)
- ``PyK()`` (CUDA kernel)
- ``PzK()`` (CUDA kernel)

