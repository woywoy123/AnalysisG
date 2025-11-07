utils.cuh
=========

**File Path**: ``src/AnalysisG/pyc/nusol/include/nusol/utils.cuh``

**File Type**: CUDA Header

**Lines**: 312

Description
-----------

_batch_idx[threadIdx.x + 2*dx] = pid[threadIdx.x][1];

Dependencies
------------

**C++ Includes**:

- ``curand.h``
- ``curand_kernel.h``
- ``math.h``
- ``utils/atomic.cuh``

CUDA Kernels
------------

- ``_assign_mass()`` (CUDA kernel)
- ``_best_sols()`` (CUDA kernel)
- ``_combination()`` (CUDA kernel)
- ``_compare_solx()`` (CUDA kernel)
- ``_count()`` (CUDA kernel)
- ``_jacobi()`` (CUDA kernel)
- ``_perturb()`` (CUDA kernel)

