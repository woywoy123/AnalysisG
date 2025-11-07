_odeRK.h
========

**File Path**: ``modules/nusol/tmp/multisol/include/multisol/_odeRK.h``

**File Type**: H (Header)

**Lines**: 66

Dependencies
------------

**Includes**:

- ``reconstruction/matrix.h``
- ``reconstruction/multisol.h``
- ``tools/tools.h``

Classes
-------

``odeRK``
~~~~~~~~~

**Inherits from**: ``tools``

**Methods**:

- ``void solve()``
- ``void rk4(double dt)``
- ``void update_t()``
- ``double solve_z_phi()``
- ``double residual(std::vector<double> wg, std::vector<double> phx)``
- ``vector<ellipse_t> derivative(const std::vector<ellipse_t>& dS)``
- ``double ghost_angle(int nui)``
- ``vector<double> plane_rk4(const std::vector<double>& t_initial)``
- ``vector<double> plane_align(const std::vector<ellipse_t>& current_state)``

Structs
-------

``ellipse_t``
~~~~~~~~~~~~~

**Members**:

- ``vec3   A,   B,   C``
- ``vec3  vA,  vB,  vC``
- ``double t``
- ``double z = 1.0``
- ``void print()``

``recon_t``
~~~~~~~~~~~

**Members**:

- ``bool is_valid = false``
- ``double residual = -1``
- ``std::vector<double> t``
- ``std::vector<double> z``
- ``std::vector<double> phi``

