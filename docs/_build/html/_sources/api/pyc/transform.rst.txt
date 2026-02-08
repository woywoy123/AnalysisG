Transform Module (PyC)
======================

Coordinate system transformations for particle physics.

Overview
--------

Located in ``pyc/pyc/transform/``, this module provides coordinate transformations 
with C++ and CUDA implementations.

Coordinate Systems
------------------

Cartesian to Polar
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pyc.transform as transform
   
   # Convert (px, py, pz, E) to (pt, eta, phi, E)
   pt, eta, phi, E = transform.cartesian_to_polar(px, py, pz, E)

Polar to Cartesian
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert (pt, eta, phi, E) to (px, py, pz, E)
   px, py, pz, E = transform.polar_to_cartesian(pt, eta, phi, E)

Lorentz Transformations
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Boost to center of mass frame
   px_cm, py_cm, pz_cm, E_cm = transform.boost(
       px, py, pz, E,
       beta_x, beta_y, beta_z
   )

Rotation Matrices
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Rotate around z-axis
   px_rot, py_rot, pz_rot = transform.rotate_z(px, py, pz, angle)

See Also
--------

* :doc:`physics` - Physics calculations
* :doc:`operators` - Tensor operations
