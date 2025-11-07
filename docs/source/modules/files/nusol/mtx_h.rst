mtx.h
=====

**File Path**: ``modules/nusol/ellipse/include/ellipse/mtx.h``

**File Type**: H (Header)

**Lines**: 71

Dependencies
------------

**Includes**:

- ``cmath``

Classes
-------

``mtx``
~~~~~~~

**Methods**:

- ``double trace()``
- ``double det()``
- ``mtx copy()``
- ``void copy(const mtx* ipt, int idx, int idy = -1)``
- ``void copy(const mtx* ipt, int idx, int jdy, int idy)``
- ``bool assign(int idx, int idy, double val, bool valid = true)``
- ``bool unique(int id1, int id2, double v1, double v2)``
- ``mtx T()``
- ``mtx inv()``
- ``mtx cof()``

