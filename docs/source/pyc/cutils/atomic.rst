atomic.cuh
==========

**File Path**: ``src/AnalysisG/pyc/cutils/include/utils/atomic.cuh``

**File Type**: CUDA Header

**Lines**: 171

Description
-----------

unsigned int idy = 4*_idy;
unsigned int idz = 4*_idz;
double ad = _M[ _x[idy  ] ][ _y[idz  ] ] * _M[ _x[idy+3] ][ _y[idz+3] ];
double bc = _M[ _x[idy+1] ][ _y[idz+1] ] * _M[ _x[idy+2] ][ _y[idz+2] ];
return (ad - bc) * _f;

Dependencies
------------

**C++ Includes**:

- ``torch/torch.h``

