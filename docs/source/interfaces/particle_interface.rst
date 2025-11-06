Particle Interface
==================

The Particle Interface provides the foundation for defining custom particle types in AnalysisG.

Overview
--------

Particles are the fundamental building blocks of events in HEP analyses. The ParticleTemplate class provides:

* Four-momentum representation and manipulation
* Particle identification (leptons, jets, etc.)
* Parent-child relationships (decay chains)
* Angular separation calculations
* Four-momentum arithmetic operations

Core ParticleTemplate Class
---------------------------

File Location
~~~~~~~~~~~~~

* **Cython Implementation**: ``src/AnalysisG/core/particle_template.pyx``
* **Cython Header**: ``src/AnalysisG/core/particle_template.pxd``

Class Definition
~~~~~~~~~~~~~~~~

.. class:: ParticleTemplate

   The base class for all particle types in AnalysisG.

Properties
~~~~~~~~~~

Four-Momentum Components
^^^^^^^^^^^^^^^^^^^^^^^^

.. property:: px
   
   Momentum x-component in GeV.
   
   :type: float

.. property:: py
   
   Momentum y-component in GeV.
   
   :type: float

.. property:: pz
   
   Momentum z-component in GeV.
   
   :type: float

.. property:: e
   
   Energy in GeV.
   
   :type: float

Kinematic Properties
^^^^^^^^^^^^^^^^^^^^

.. property:: pt
   
   Transverse momentum in GeV.
   
   :type: float

.. property:: eta
   
   Pseudorapidity.
   
   :type: float

.. property:: phi
   
   Azimuthal angle in radians.
   
   :type: float

.. property:: mass
   
   Invariant mass in GeV.
   
   :type: float

Particle Identification
^^^^^^^^^^^^^^^^^^^^^^^

.. property:: pdgid
   
   PDG particle identification code.
   
   :type: int

.. property:: charge
   
   Electric charge in units of elementary charge.
   
   :type: float

Decay Chain
^^^^^^^^^^^

.. property:: Parents
   
   List of parent particles in the decay chain.
   
   :type: list[ParticleTemplate]

.. property:: Children
   
   List of child particles in the decay chain.
   
   :type: list[ParticleTemplate]

Methods to Override
~~~~~~~~~~~~~~~~~~~

Particle Type Identification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: is_lepton() -> bool
   
   Determine if particle is a lepton (e, μ, τ).
   
   :returns: True if particle is a lepton
   :rtype: bool

.. method:: is_jet() -> bool
   
   Determine if particle is a jet.
   
   :returns: True if particle is a jet
   :rtype: bool

.. method:: is_bjet() -> bool
   
   Determine if particle is a b-tagged jet.
   
   :returns: True if particle is a b-jet
   :rtype: bool

Utility Methods
~~~~~~~~~~~~~~~

.. method:: DeltaR(ParticleTemplate other) -> float
   
   Calculate angular separation (ΔR) from another particle.
   
   :param other: Other particle
   :type other: ParticleTemplate
   :returns: Angular separation ΔR = √(Δη² + Δφ²)
   :rtype: float

.. method:: clone() -> ParticleTemplate
   
   Create a deep copy of this particle.
   
   :returns: Cloned particle
   :rtype: ParticleTemplate

Operator Overloads
~~~~~~~~~~~~~~~~~~

Four-Momentum Addition
^^^^^^^^^^^^^^^^^^^^^^

.. method:: __add__(ParticleTemplate other) -> ParticleTemplate
   
   Add four-momenta of two particles.
   
   :param other: Particle to add
   :type other: ParticleTemplate
   :returns: New particle with combined four-momentum
   :rtype: ParticleTemplate

.. method:: __iadd__(ParticleTemplate other) -> ParticleTemplate
   
   In-place four-momentum addition.
   
   :param other: Particle to add
   :type other: ParticleTemplate
   :returns: Self with updated four-momentum
   :rtype: ParticleTemplate

Usage Examples
--------------

Basic Particle Selection
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import ParticleTemplate
   
   class MyParticle(ParticleTemplate):
       def is_lepton(self):
           # Electrons and muons
           return abs(self.pdgid) in [11, 13]
       
       def is_jet(self):
           # Jets typically have pdgid of 0 or large values
           return self.pdgid == 0

Custom Particle Type
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import ParticleTemplate
   
   class Electron(ParticleTemplate):
       def __init__(self, data=None):
           super().__init__(data)
           self.isolation = 0.0
       
       def is_lepton(self):
           return True
       
       def is_isolated(self):
           return self.isolation < 0.1

Four-Momentum Operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Combine particles
   top_candidate = lepton + jet1 + jet2
   
   # Check invariant mass
   if 150 < top_candidate.mass < 200:
       print(f"Top candidate found: m = {top_candidate.mass:.1f} GeV")
   
   # Calculate separation
   dr = lepton.DeltaR(jet1)
   if dr > 0.4:
       print("Particles are well separated")

Best Practices
--------------

1. **Preserve immutability**: Use + operator for new particles, += for in-place modification.

2. **Check PDG IDs carefully**: Different experiments may use different conventions.

3. **Handle missing data**: Check for None or invalid values before calculations.

4. **Use DeltaR for isolation**: Common pattern for checking particle isolation.

See Also
--------

* :doc:`event_interface`: Event interface documentation
* :doc:`../core/particle_template`: Core ParticleTemplate implementation
