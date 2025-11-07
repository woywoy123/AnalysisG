base.cuh
========

**File Path**: ``src/AnalysisG/pyc/physics/include/physics/base.cuh``

**File Type**: CUDA Header

**Lines**: 201

Description
-----------

const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x;
sdata[threadIdx.x][threadIdx.y] = pn*pn;

Dependencies
------------

**C++ Includes**:

- ``utils/atomic.cuh``

CUDA Kernels
------------

- ``_Beta()`` (CUDA kernel)
- ``_Beta2()`` (CUDA kernel)
- ``_M()`` (CUDA kernel)
- ``_M2()`` (CUDA kernel)
- ``_Mt()`` (CUDA kernel)
- ``_Mt2()`` (CUDA kernel)
- ``_P2K()`` (CUDA kernel)
- ``_PK()`` (CUDA kernel)
- ``_deltar()`` (CUDA kernel)
- ``_theta()`` (CUDA kernel)

