ellipse.h
=========

**File Path**: ``modules/nusol/ellipse/include/ellipse/ellipse.h``

**File Type**: H (Header)

**Lines**: 69

Dependencies
------------

**Includes**:

- ``ellipse/mtx.h``
- ``ellipse/nusol.h``
- ``ellipse/solvers.h``
- ``templates/particle_template.h``

Classes
-------

``ellipse``
~~~~~~~~~~~

**Methods**:

- ``void prepare(double mt, double mw)``
- ``vector<particle_template*> nunu_make()``
- ``void solve()``
- ``void flush()``
- ``void make_neutrinos(mtx* v, mtx* v_)``
- ``int generate(nuelx* nu1, nuelx* nu2)``
- ``int intersection(mtx** v, mtx** v_)``
- ``int angle_cross(mtx** v, mtx** v_)``

Structs
-------

``nusol_t``
~~~~~~~~~~~

``nunu_t``
~~~~~~~~~~

**Members**:

- ``nunu_t()``
- ``~nunu_t()``
- ``mtx* nu1 = nullptr``
- ``mtx* nu2 = nullptr``
- ``mtx* agl = nullptr``
- ``nuelx* nux1 = nullptr``
- ``nuelx* nux2 = nullptr``

