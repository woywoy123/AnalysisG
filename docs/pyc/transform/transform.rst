transform Module
===============

The ``transform`` module provides functions for coordinate transformations commonly used in particle physics.

Coordinate Conversions
--------------------

.. py:function:: cartesian_to_polar(x, y, z=None)

   Convert cartesian coordinates to polar coordinates.
   
   :param torch.Tensor x: x-coordinate tensor
   :param torch.Tensor y: y-coordinate tensor
   :param torch.Tensor z: z-coordinate tensor (optional)
   :return: If z is None: (r, phi), else: (r, theta, phi)
   :rtype: tuple of torch.Tensor

.. py:function:: polar_to_cartesian(r, theta=None, phi=None)

   Convert polar coordinates to cartesian coordinates.
   
   :param torch.Tensor r: radial distance tensor
   :param torch.Tensor theta: polar angle tensor (optional)
   :param torch.Tensor phi: azimuthal angle tensor
   :return: If theta is None: (x, y), else: (x, y, z)
   :rtype: tuple of torch.Tensor

Pseudorapidity Functions
----------------------

.. py:function:: eta_from_theta(theta)

   Calculate pseudorapidity from polar angle.
   
   :param torch.Tensor theta: polar angle tensor
   :return: pseudorapidity tensor
   :rtype: torch.Tensor

.. py:function:: theta_from_eta(eta)

   Calculate polar angle from pseudorapidity.
   
   :param torch.Tensor eta: pseudorapidity tensor
   :return: polar angle tensor
   :rtype: torch.Tensor

.. py:function:: eta_from_pz_pt(pz, pt)

   Calculate pseudorapidity from longitudinal momentum and transverse momentum.
   
   :param torch.Tensor pz: longitudinal momentum tensor
   :param torch.Tensor pt: transverse momentum tensor
   :return: pseudorapidity tensor
   :rtype: torch.Tensor

Angular Functions
--------------

.. py:function:: delta_phi(phi1, phi2)

   Calculate the delta phi between two azimuthal angles, keeping the result between -π and π.
   
   :param torch.Tensor phi1: first azimuthal angle tensor
   :param torch.Tensor phi2: second azimuthal angle tensor
   :return: delta phi tensor
   :rtype: torch.Tensor

.. py:function:: delta_r(eta1, phi1, eta2, phi2)

   Calculate the ΔR separation between two points in η-φ space.
   
   :param torch.Tensor eta1: first pseudorapidity tensor
   :param torch.Tensor phi1: first azimuthal angle tensor
   :param torch.Tensor eta2: second pseudorapidity tensor
   :param torch.Tensor phi2: second azimuthal angle tensor
   :return: ΔR separation tensor
   :rtype: torch.Tensor

Examples
-------

Basic usage examples:

.. code-block:: python

   import torch
   from AnalysisG.pyc import transform
   
   # Create some input tensors
   x = torch.tensor([1.0, 2.0, 3.0])
   y = torch.tensor([1.0, 2.0, 3.0])
   z = torch.tensor([1.0, 2.0, 3.0])
   
   # Convert from cartesian to polar
   r, theta, phi = transform.cartesian_to_polar(x, y, z)
   print(f"r: {r}")
   print(f"theta: {theta}")
   print(f"phi: {phi}")
   
   # Convert from polar back to cartesian
   x_new, y_new, z_new = transform.polar_to_cartesian(r, theta, phi)
   
   # Calculate pseudorapidity from polar angle
   eta = transform.eta_from_theta(theta)
   
   # Calculate delta R between two points
   eta1 = torch.tensor([0.0, 1.0, 2.0])
   phi1 = torch.tensor([0.0, 0.5, 1.0])
   eta2 = torch.tensor([0.1, 1.2, 2.5])
   phi2 = torch.tensor([0.2, 0.7, 1.5])
   
   dr = transform.delta_r(eta1, phi1, eta2, phi2)
   print(f"Delta R: {dr}")