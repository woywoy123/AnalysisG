multisol.h
==========

**File Path**: ``modules/nusol/tmp/multisol/include/multisol/multisol.h``

**File Type**: H (Header)

**Lines**: 143

Dependencies
------------

**Includes**:

- ``iostream``
- ``reconstruction/matrix.h``
- ``string``
- ``templates/particle_template.h``

Classes
-------

``multisol``
~~~~~~~~~~~~

**Methods**:

- ``void make_rt()``
- ``matrix H_tilde(double t, double z)``
- ``matrix dHdt_tilde(double t, double z = 1)``
- ``matrix H(double t, double z)``
- ``matrix dHdt(double t, double z = 1)``
- ``matrix d2Hdt2(double t, double z = 1)``
- ``vec3 v(double t, double z, double phi)``
- ``vec3 dv_dt(double t, double z, double phi)``
- ``vec3 dv_dphi(double t, double z, double phi)``
- ``vec3 d2v_dt_dphi(double t, double z, double phi)``

Structs
-------

``multisol_t``
~~~~~~~~~~~~~~

**Members**:

- ``double A = 0, B = 0, C = 0``
- ``double D = 0, E = 0, F = 0``

``rev_t``
~~~~~~~~~

**Members**:

- ``double u = 0``
- ``double v = 0``
- ``double u_p = 0``
- ``double v_p = 0``
- ``double t = 0``
- ``double z = 0``

``mass_t``
~~~~~~~~~~

**Members**:

- ``double mt = 0``
- ``double mw = 0``
- ``double sx = 0``
- ``double sy = 0``

``geo_t``
~~~~~~~~~

**Members**:

- ``~geo_t()``
- ``double px(double u, double v)``
- ``double py(double u, double v)``
- ``double pz(double u, double v)``
- ``vec3 r0, d, _pts1, _pts2``
- ``double lx(double s)``
- ``double ly(double s)``
- ``double lz(double s)``

