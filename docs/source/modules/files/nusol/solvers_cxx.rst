solvers.cxx
===========

**File Path**: ``modules/nusol/ellipse/cxx/solvers.cxx``

**File Type**: CXX (Source)

**Lines**: 515

Dependencies
------------

**Includes**:

- ``cmath``
- ``complex``
- ``ellipse/mtx.h``
- ``ellipse/solvers.h``
- ``iostream``

Functions
---------

``mtx make_ellipse(mtx* H, double angle)``

``double distance(mtx* H1, double a1, mtx* H2, double a2)``

``void swap_index(double** v, int idx)``

``void multisqrt(double y, double roots[2], int *count)``

``void factor_degenerate(mtx G, mtx* lines, int* lc, double* q0)``

``int intersections_ellipse_line(mtx* ellipse, mtx* line, mtx* pts)``

``int intersection_ellipses(mtx* A, mtx* B, mtx** lines, mtx** pts, mtx** sols)``

``void _arith(double** o, double** v2, double s, int idx, int idy)``

``void _scale(double** v, double** f, int idx, int idy, double s)``

``void _copy(double* dst, double* src, int lx)``

``void _copy(bool* dst, bool* src, int lx)``

``void _copy(double** dst, double** src, int lx, int ly)``

``void _copy(bool** dst, bool** src, int lx, int ly)``

``double _trace(double** A)``

``double _m_00(double**  M)``

