Physics Module (PyC)
====================

The PyC physics module provides high-performance physics calculations.

Overview
--------

Located in ``pyc/pyc/physics/``, this module implements common particle physics calculations 
with optimized C++ and CUDA backends.

API Reference
-------------

Delta R Calculation
~~~~~~~~~~~~~~~~~~~

.. function:: DeltaR(eta1, phi1, eta2, phi2)

   Calculate angular distance (ΔR) between two particles.

   :param eta1: Pseudorapidity of first particle
   :param phi1: Azimuthal angle of first particle (radians)
   :param eta2: Pseudorapidity of second particle
   :param phi2: Azimuthal angle of second particle (radians)
   :return: ΔR = √(Δη² + Δφ²)
   :rtype: float or torch.Tensor

   .. math::

      \Delta R = \sqrt{(\eta_1 - \eta_2)^2 + (\phi_1 - \phi_2)^2}

   Example:

   .. code-block:: python

      import pyc.physics as phys
      
      dr = phys.DeltaR(jet_eta, jet_phi, lepton_eta, lepton_phi)

Invariant Mass
~~~~~~~~~~~~~~

.. function:: InvariantMass(pt1, eta1, phi1, E1, pt2, eta2, phi2, E2)

   Calculate invariant mass of two particles.

   :param pt1: Transverse momentum of first particle [GeV]
   :param eta1: Pseudorapidity of first particle
   :param phi1: Azimuthal angle of first particle
   :param E1: Energy of first particle [GeV]
   :param pt2: Transverse momentum of second particle [GeV]
   :param eta2: Pseudorapidity of second particle
   :param phi2: Azimuthal angle of second particle
   :param E2: Energy of second particle [GeV]
   :return: Invariant mass [GeV]
   :rtype: float or torch.Tensor

   .. math::

      m_{inv} = \sqrt{(E_1 + E_2)^2 - |\vec{p}_1 + \vec{p}_2|^2}

   Example:

   .. code-block:: python

      mass = phys.InvariantMass(
          jet1.pt, jet1.eta, jet1.phi, jet1.E,
          jet2.pt, jet2.eta, jet2.phi, jet2.E
      )

Transverse Mass
~~~~~~~~~~~~~~~

.. function:: TransverseMass(pt1, phi1, pt2, phi2)

   Calculate transverse mass.

   :param pt1: Transverse momentum of first particle [GeV]
   :param phi1: Azimuthal angle of first particle
   :param pt2: Transverse momentum of second particle [GeV]
   :param phi2: Azimuthal angle of second particle
   :return: Transverse mass [GeV]
   :rtype: float or torch.Tensor

   .. math::

      m_T = \sqrt{2 p_{T,1} p_{T,2} (1 - \cos(\phi_1 - \phi_2))}

Four-Vector Operations
~~~~~~~~~~~~~~~~~~~~~~

.. function:: FourVector(pt, eta, phi, E)

   Create a 4-vector from kinematic variables.

   :param pt: Transverse momentum [GeV]
   :param eta: Pseudorapidity
   :param phi: Azimuthal angle
   :param E: Energy [GeV]
   :return: 4-vector (px, py, pz, E)
   :rtype: tuple or torch.Tensor

.. function:: AddFourVectors(vec1, vec2)

   Add two 4-vectors.

   :param vec1: First 4-vector
   :param vec2: Second 4-vector
   :return: Sum 4-vector
   :rtype: torch.Tensor

Angular Utilities
~~~~~~~~~~~~~~~~~

.. function:: DeltaPhi(phi1, phi2)

   Calculate difference in azimuthal angle, accounting for 2π periodicity.

   :param phi1: First angle [radians]
   :param phi2: Second angle [radians]
   :return: Δφ in range [-π, π]
   :rtype: float or torch.Tensor

.. function:: DeltaEta(eta1, eta2)

   Calculate difference in pseudorapidity.

   :param eta1: First pseudorapidity
   :param eta2: Second pseudorapidity
   :return: Δη
   :rtype: float or torch.Tensor

Batch Operations
----------------

All functions support batch operations with tensors:

.. code-block:: python

   import torch
   import pyc.physics as phys
   
   # Batch of 1000 particle pairs
   eta1 = torch.randn(1000)
   phi1 = torch.randn(1000)
   eta2 = torch.randn(1000)
   phi2 = torch.randn(1000)
   
   # Compute all delta R values at once
   dr = phys.DeltaR(eta1, phi1, eta2, phi2)  # shape: (1000,)

GPU Acceleration
----------------

Operations automatically use GPU when tensors are on GPU:

.. code-block:: python

   import torch
   import pyc.physics as phys
   
   # Move to GPU
   eta1_gpu = eta1.cuda()
   phi1_gpu = phi1.cuda()
   eta2_gpu = eta2.cuda()
   phi2_gpu = phi2.cuda()
   
   # Computation runs on GPU
   dr_gpu = phys.DeltaR(eta1_gpu, phi1_gpu, eta2_gpu, phi2_gpu)

Implementation
--------------

The physics module is implemented in:

- ``physics.cxx``: C++ CPU implementation
- ``include/physics/physics.h``: C++ header
- ``cuda/physics.cxx``: CUDA GPU implementation

Performance
-----------

Typical speedups over numpy (CPU):

- Delta R: 50-100x faster
- Invariant mass: 30-80x faster
- Batch operations: 100-1000x faster on GPU

See Also
--------

* :doc:`operators` - Tensor operations
* :doc:`transform` - Coordinate transformations
