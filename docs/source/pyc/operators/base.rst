base.cuh
========

**File Path**: ``src/AnalysisG/pyc/operators/include/operators/base.cuh``

**File Type**: CUDA Header

**Lines**: 300

Description
-----------

const unsigned int idx  = blockIdx.x * blockDim.x + threadIdx.x;

Dependencies
------------

**C++ Includes**:

- ``utils/atomic.cuh``

CUDA Kernels
------------

- ``_cofactor()`` (CUDA kernel)
- ``_costheta()`` (CUDA kernel)
- ``_cross()`` (CUDA kernel)
- ``_determinant()`` (CUDA kernel)
- ``_dot()`` (CUDA kernel)
- ``_eigenvalue()`` (CUDA kernel)
- ``_inverse()`` (CUDA kernel)
- ``_rt()`` (CUDA kernel)
- ``_rx()`` (CUDA kernel)
- ``_ry()`` (CUDA kernel)
- ``_rz()`` (CUDA kernel)

