physics Module
==============

The ``physics`` module provides essential physics calculations for particle physics analysis.

Energy and Momentum Functions
---------------------------

.. py:function:: energy(p, m)

   Calculate energy using momentum and mass.
   
   :param torch.Tensor p: momentum tensor (px, py, pz) or magnitude of momentum
   :param torch.Tensor m: mass tensor
   :return: energy tensor
   :rtype: torch.Tensor

.. py:function:: momentum(e, m)

   Calculate momentum magnitude using energy and mass.
   
   :param torch.Tensor e: energy tensor
   :param torch.Tensor m: mass tensor
   :return: momentum magnitude tensor
   :rtype: torch.Tensor

.. py:function:: transverse_momentum(px, py)

   Calculate transverse momentum from momentum components.
   
   :param torch.Tensor px: x-component of momentum
   :param torch.Tensor py: y-component of momentum
   :return: transverse momentum tensor
   :rtype: torch.Tensor

.. py:function:: transverse_energy(et, eta)

   Calculate transverse energy from energy and pseudorapidity.
   
   :param torch.Tensor e: energy tensor
   :param torch.Tensor eta: pseudorapidity tensor
   :return: transverse energy tensor
   :rtype: torch.Tensor

Mass Functions
--------------

.. py:function:: invariant_mass(e, px, py, pz)

   Calculate invariant mass from four-momentum components.
   
   :param torch.Tensor e: energy tensor
   :param torch.Tensor px: x-component of momentum
   :param torch.Tensor py: y-component of momentum
   :param torch.Tensor pz: z-component of momentum
   :return: invariant mass tensor
   :rtype: torch.Tensor

.. py:function:: invariant_mass_squared(e, px, py, pz)

   Calculate squared invariant mass from four-momentum components.
   
   :param torch.Tensor e: energy tensor
   :param torch.Tensor px: x-component of momentum
   :param torch.Tensor py: y-component of momentum
   :param torch.Tensor pz: z-component of momentum
   :return: squared invariant mass tensor
   :rtype: torch.Tensor

.. py:function:: transverse_mass(et1, phi1, et2, phi2)

   Calculate transverse mass between two particles.
   
   :param torch.Tensor et1: transverse energy of first particle
   :param torch.Tensor phi1: azimuthal angle of first particle
   :param torch.Tensor et2: transverse energy of second particle
   :param torch.Tensor phi2: azimuthal angle of second particle
   :return: transverse mass tensor
   :rtype: torch.Tensor

Relativistic Functions
-------------------

.. py:function:: beta(p, e)

   Calculate relativistic beta (v/c) from momentum and energy.
   
   :param torch.Tensor p: momentum tensor
   :param torch.Tensor e: energy tensor
   :return: relativistic beta tensor
   :rtype: torch.Tensor

.. py:function:: gamma(beta=None, e=None, m=None)

   Calculate relativistic gamma factor.
   
   :param torch.Tensor beta: relativistic beta tensor (optional)
   :param torch.Tensor e: energy tensor (optional, requires m)
   :param torch.Tensor m: mass tensor (optional, requires e)
   :return: relativistic gamma tensor
   :rtype: torch.Tensor

Examples
-------

Basic usage examples:

.. code-block:: python

   import torch
   from AnalysisG.pyc import physics
   
   # Create some input tensors
   px = torch.tensor([10.0, 20.0, 30.0])
   py = torch.tensor([15.0, 25.0, 35.0])
   pz = torch.tensor([20.0, 30.0, 40.0])
   m = torch.tensor([0.5, 1.0, 1.5])
   
   # Calculate magnitude of momentum
   p = torch.sqrt(px**2 + py**2 + pz**2)
   
   # Calculate energy
   e = physics.energy(p, m)
   print(f"Energy: {e}")
   
   # Calculate transverse momentum
   pt = physics.transverse_momentum(px, py)
   print(f"Transverse momentum: {pt}")
   
   # Calculate invariant mass
   mass = physics.invariant_mass(e, px, py, pz)
   print(f"Invariant mass: {mass}")
   
   # Calculate relativistic beta
   beta = physics.beta(p, e)
   print(f"Relativistic beta: {beta}")
   
   # Calculate relativistic gamma
   gamma = physics.gamma(beta=beta)
   print(f"Relativistic gamma: {gamma}")
   
   # Alternative way to calculate gamma
   gamma_alt = physics.gamma(e=e, m=m)
   print(f"Relativistic gamma (alternative): {gamma_alt}")