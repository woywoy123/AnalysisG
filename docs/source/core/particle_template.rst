ParticleTemplate
================

The ParticleTemplate class represents individual particles in high-energy physics events. It provides four-momentum manipulation, particle identification, and decay chain tracking.

**File Location**: ``src/AnalysisG/core/particle_template.pyx``

Overview
--------

ParticleTemplate is the base class for all particle types in AnalysisG. It provides:

* Four-momentum representation (px, py, pz, E)
* Kinematic properties (pt, eta, phi, mass)
* Particle identification (PDG ID, charge)
* Parent-child relationships for decay chains
* Four-momentum arithmetic operations
* Angular separation calculations (ΔR)
* Serialization and cloning

The class uses a C++ backend (``particle_template``) for performance-critical operations.

Kinematic Properties
--------------------

Four-Momentum Components
~~~~~~~~~~~~~~~~~~~~~~~~

.. py:attribute:: px
   
   Momentum x-component in MeV.
   
   :type: float
   :getter: Returns px

.. py:attribute:: py
   
   Momentum y-component in MeV.
   
   :type: float
   :getter: Returns py

.. py:attribute:: pz
   
   Momentum z-component in MeV.
   
   :type: float
   :getter: Returns pz

.. py:attribute:: e
   
   Energy in MeV.
   
   :type: float
   :getter: Returns energy

Derived Kinematic Quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:attribute:: pt
   
   Transverse momentum in MeV.
   
   :type: float
   :getter: Returns pt = √(px² + py²)
   
   **Example**:
   
   .. code-block:: python
   
      if particle.pt > 25e3:  # 25 GeV
          print(f"High-pT particle: {particle.pt/1e3:.1f} GeV")

.. py:attribute:: eta
   
   Pseudorapidity.
   
   :type: float
   :getter: Returns η = -ln(tan(θ/2))
   
   **Definition**: Related to polar angle θ by η = -ln(tan(θ/2))
   
   **Range**: -∞ to +∞ (forward → large positive, backward → large negative)
   
   **Example**:
   
   .. code-block:: python
   
      # Central detector acceptance
      if abs(particle.eta) < 2.5:
          print(f"Central particle: η = {particle.eta:.2f}")

.. py:attribute:: phi
   
   Azimuthal angle in radians.
   
   :type: float
   :getter: Returns φ = atan2(py, px)
   
   **Range**: -π to +π
   
   **Example**:
   
   .. code-block:: python
   
      import math
      phi_deg = particle.phi * 180 / math.pi
      print(f"φ = {phi_deg:.1f}°")

.. py:attribute:: Mass
   
   Invariant mass in MeV.
   
   :type: float
   :getter: Returns m = √(E² - p²)
   
   **Note**: Can be negative for off-shell particles
   
   **Example**:
   
   .. code-block:: python
   
      # Check if mass is consistent with a top quark
      if 150e3 < particle.Mass < 200e3:
          print(f"Top-like mass: {particle.Mass/1e3:.1f} GeV")

Particle Identification
-----------------------

.. py:attribute:: pdgid
   
   PDG particle identification code.
   
   :type: int
   :getter: Returns PDG ID
   
   **Common PDG IDs**:
   
   * Photon: 22
   * Electron: 11 (positron: -11)
   * Muon: 13 (antimuon: -13)
   * Tau: 15 (antitau: -15)
   * Neutrinos: 12 (νe), 14 (νμ), 16 (ντ)
   * Up quark: 2, Down: 1, Strange: 3, Charm: 4, Bottom: 5, Top: 6
   * W boson: 24 (W-: -24)
   * Z boson: 23
   * Gluon: 21
   * Higgs: 25
   
   **Example**:
   
   .. code-block:: python
   
      # Identify particle type
      if abs(particle.pdgid) == 11:
          print("Electron/positron")
      elif abs(particle.pdgid) == 13:
          print("Muon/antimuon")
      elif abs(particle.pdgid) == 6:
          print("Top/antitop quark")

.. py:attribute:: charge
   
   Electric charge in units of elementary charge.
   
   :type: float
   :getter: Returns charge
   
   **Example**:
   
   .. code-block:: python
   
      # Check charge
      if particle.charge > 0:
          print("Positively charged")
      elif particle.charge < 0:
          print("Negatively charged")
      else:
          print("Neutral")

.. py:attribute:: symbol
   
   Particle symbol/name.
   
   :type: str
   :getter: Returns particle symbol (e.g., "e-", "μ+", "t")

.. py:attribute:: Type
   
   Particle type identifier.
   
   :type: str
   :getter: Returns type string

Decay Chain Information
-----------------------

.. py:attribute:: Parents
   
   List of parent particles in the decay chain.
   
   :type: list[ParticleTemplate]
   :getter: Returns list of parents
   :setter: Sets parent list
   
   **Purpose**: Track which particles decayed to produce this particle.
   
   **Example**:
   
   .. code-block:: python
   
      # Find particles from top decays
      for particle in event.Particles:
          for parent in particle.Parents:
              if abs(parent.pdgid) == 6:  # Top quark
                  print(f"Particle from top decay: {particle.symbol}")

.. py:attribute:: Children
   
   List of child particles (decay products).
   
   :type: list[ParticleTemplate]
   :getter: Returns list of children
   :setter: Sets children list
   
   **Purpose**: Track decay products of this particle.
   
   **Example**:
   
   .. code-block:: python
   
      # Examine W boson decays
      for particle in event.Particles:
          if abs(particle.pdgid) == 24:  # W boson
              print(f"W → ", end="")
              for child in particle.Children:
                  print(f"{child.symbol} ", end="")
              print()

Identification Methods
---------------------

.. py:attribute:: index
   
   Particle index in the event.
   
   :type: int
   :getter: Returns index

.. py:attribute:: hash
   
   Unique hash identifier for this particle.
   
   :type: str
   :getter: Returns hash string

Core Methods
------------

DeltaR()
~~~~~~~~

.. py:method:: DeltaR(other) -> float
   
   Calculate angular separation (ΔR) from another particle.
   
   :param other: Another ParticleTemplate instance
   :type other: ParticleTemplate
   :returns: ΔR = √(Δη² + Δφ²)
   :rtype: float
   
   **Purpose**: Measure angular distance between particles in (η, φ) space.
   
   **Applications**:
   
   * Jet/lepton overlap removal
   * Isolation requirements
   * Object matching
   * Decay topology studies
   
   **Example**:
   
   .. code-block:: python
   
      # Overlap removal: remove jets too close to leptons
      leptons = [p for p in event.Particles if p.is_lepton()]
      jets = [p for p in event.Particles if p.is_jet()]
      
      clean_jets = []
      for jet in jets:
          is_isolated = True
          for lepton in leptons:
              if jet.DeltaR(lepton) < 0.4:  # 0.4 overlap cone
                  is_isolated = False
                  break
          if is_isolated:
              clean_jets.append(jet)

clone()
~~~~~~~

.. py:method:: clone() -> ParticleTemplate
   
   Create a deep copy of this particle.
   
   :returns: New ParticleTemplate instance with same properties
   :rtype: ParticleTemplate
   
   **Example**:
   
   .. code-block:: python
   
      original = event.Particles[0]
      copy = original.clone()
      copy.pt  # Same as original, but independent object

is_self()
~~~~~~~~~

.. py:method:: is_self(inpt) -> bool
   
   Check if input is a ParticleTemplate or subclass.
   
   :param inpt: Object to check
   :returns: True if inpt is ParticleTemplate-derived
   :rtype: bool

Arithmetic Operations
--------------------

Four-Momentum Addition
~~~~~~~~~~~~~~~~~~~~~~

ParticleTemplate supports four-momentum addition for combining particles:

.. py:method:: __add__(other) -> ParticleTemplate
   
   Add four-momenta of two particles (creates new particle).
   
   :param other: Particle to add
   :type other: ParticleTemplate
   :returns: New particle with combined four-momentum
   :rtype: ParticleTemplate
   
   **Physics**: Adds (px, py, pz, E) component-wise to create composite particle.
   
   **Example - W Boson Reconstruction**:
   
   .. code-block:: python
   
      # Find opposite-sign leptons
      electrons = [p for p in event.Particles 
                   if abs(p.pdgid) == 11 and p.pt > 25e3]
      
      for i, e1 in enumerate(electrons):
          for e2 in electrons[i+1:]:
              if e1.charge * e2.charge < 0:  # Opposite sign
                  # Combine to form W candidate
                  W_candidate = e1 + e2
                  print(f"W mass: {W_candidate.Mass/1e3:.1f} GeV")
                  print(f"W pt: {W_candidate.pt/1e3:.1f} GeV")

.. py:method:: __iadd__(other) -> ParticleTemplate
   
   In-place four-momentum addition.
   
   :param other: Particle to add
   :type other: ParticleTemplate
   :returns: Self with updated four-momentum
   :rtype: ParticleTemplate
   
   **Example**:
   
   .. code-block:: python
   
      # Build composite particle step-by-step
      top = lepton.clone()
      top += b_jet
      top += missing_et  # Add neutrino approximation

Comparison Operators
~~~~~~~~~~~~~~~~~~~

.. py:method:: __eq__(other) -> bool
   
   Compare particles for equality.
   
   :param other: Another particle
   :returns: True if particles have same four-momentum and properties
   :rtype: bool

.. py:method:: __hash__() -> int
   
   Return integer hash for dictionary/set usage.
   
   :returns: Integer hash
   :rtype: int

Complete Usage Examples
-----------------------

Example 1: Lepton Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def select_leptons(event, pt_min=25e3, eta_max=2.5):
       """Select good quality leptons."""
       leptons = []
       
       for particle in event.Particles:
           # Check if it's a lepton
           if abs(particle.pdgid) not in [11, 13]:
               continue
           
           # Apply kinematic cuts
           if particle.pt < pt_min:
               continue
           
           if abs(particle.eta) > eta_max:
               continue
           
           leptons.append(particle)
       
       return leptons

Example 2: Top Quark Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def reconstruct_tops(event):
       """Reconstruct top quarks from decay products."""
       # Get particle collections
       leptons = [p for p in event.Particles 
                  if abs(p.pdgid) in [11, 13] and p.pt > 25e3]
       jets = [p for p in event.Particles 
               if p.is_jet() and p.pt > 30e3]
       bjets = [j for j in jets if j.is_bjet()]
       
       top_candidates = []
       
       # Try all combinations of lepton + b-jet
       for lepton in leptons:
           for bjet in bjets:
               # Check angular separation
               if lepton.DeltaR(bjet) < 0.4:
                   continue  # Too close, likely same object
               
               # Combine four-momenta
               top = lepton + bjet
               
               # Apply mass window cut
               if 150e3 < top.Mass < 200e3:  # 150-200 GeV
                   top_candidates.append({
                       'top': top,
                       'lepton': lepton,
                       'bjet': bjet,
                       'mass': top.Mass / 1e3,  # GeV
                       'pt': top.pt / 1e3        # GeV
                   })
       
       return top_candidates

Example 3: Jet Cleaning (Overlap Removal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def clean_jets(jets, leptons, dr_threshold=0.4):
       """Remove jets overlapping with leptons."""
       clean_jets = []
       
       for jet in jets:
           # Check distance to all leptons
           overlaps = False
           for lepton in leptons:
               if jet.DeltaR(lepton) < dr_threshold:
                   overlaps = True
                   break
           
           # Keep jet if it doesn't overlap
           if not overlaps:
               clean_jets.append(jet)
       
       return clean_jets

Example 4: Invariant Mass Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def calculate_invariant_mass(particles):
       """Calculate invariant mass of particle system."""
       if len(particles) == 0:
           return 0.0
       
       # Sum four-momenta
       total = particles[0].clone()
       for particle in particles[1:]:
           total += particle
       
       return total.Mass

   # Usage: Z → ee mass
   electrons = [p for p in event.Particles if abs(p.pdgid) == 11]
   if len(electrons) >= 2:
       Z_mass = calculate_invariant_mass(electrons[:2])
       print(f"Z candidate mass: {Z_mass/1e3:.1f} GeV")

Example 5: Decay Chain Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def find_top_decay_products(event):
       """Find particles from top quark decays."""
       top_products = {
           'W_bosons': [],
           'b_quarks': [],
           'leptons': [],
           'neutrinos': []
       }
       
       # Find all particles
       for particle in event.Particles:
           # Check parents
           for parent in particle.Parents:
               if abs(parent.pdgid) == 6:  # Parent is top quark
                   # Categorize decay product
                   if abs(particle.pdgid) == 24:
                       top_products['W_bosons'].append(particle)
                   elif abs(particle.pdgid) == 5:
                       top_products['b_quarks'].append(particle)
                   elif abs(particle.pdgid) in [11, 13, 15]:
                       top_products['leptons'].append(particle)
                   elif abs(particle.pdgid) in [12, 14, 16]:
                       top_products['neutrinos'].append(particle)
       
       return top_products

Common Patterns and Best Practices
-----------------------------------

Kinematic Cuts
~~~~~~~~~~~~~~

Always apply cuts in the most efficient order (most restrictive first):

.. code-block:: python

   def passes_kinematic_cuts(particle):
       """Check if particle passes kinematic selection."""
       # Most restrictive first for performance
       if particle.pt < 25e3:  # 25 GeV - fastest check
           return False
       
       if abs(particle.eta) > 2.5:  # Detector acceptance
           return False
       
       return True

Unit Consistency
~~~~~~~~~~~~~~~~

ROOT typically uses MeV. Be explicit about units:

.. code-block:: python

   # Good: explicit units
   if particle.pt > 25e3:  # 25 GeV in MeV
       pass
   
   # Also good: comment units
   pt_gev = particle.pt / 1e3  # Convert to GeV
   if pt_gev > 25:
       pass

Null Checks
~~~~~~~~~~~

Always check for empty lists before processing:

.. code-block:: python

   # Bad: can crash
   leading_jet = jets[0]
   
   # Good: safe
   if len(jets) > 0:
       leading_jet = jets[0]
   
   # Better: explicit check
   leading_jet = jets[0] if jets else None

Physics Constants
~~~~~~~~~~~~~~~~~

Define physics constants explicitly:

.. code-block:: python

   # Particle masses (MeV)
   W_MASS = 80379.0
   Z_MASS = 91187.6
   TOP_MASS = 172760.0
   
   # Mass windows (MeV)
   W_WINDOW = (60e3, 100e3)
   TOP_WINDOW = (150e3, 200e3)
   
   # Apply cuts
   if W_WINDOW[0] < W_candidate.Mass < W_WINDOW[1]:
       print("W boson candidate found")

See Also
--------

* :doc:`event_template`: EventTemplate documentation
* :doc:`../interfaces/particle_interface`: Particle interface details
* :doc:`../events/overview`: Event implementations using ParticleTemplate
