EventTemplate
==============

The EventTemplate class is the base class for all event-level data structures in AnalysisG. It represents a single physics event and provides the interface for event selection, processing, and data access.

**File Location**: ``src/AnalysisG/core/event_template.pyx``

Overview
--------

EventTemplate is designed to be inherited by user-defined event classes. It handles:

* Reading ROOT tree data (tree names, branches)
* Managing particle collections within the event
* Event weighting for statistical analysis
* Event-level selection logic
* Serialization and metadata management

The class uses a C++ backend for performance while exposing a Python/Cython interface for ease of use.

Class Hierarchy
---------------

.. code-block:: text

   EventTemplate (base class)
   └── BSM4Tops (concrete implementation)
   └── ExpMC20 (concrete implementation)
   └── SSLMC20 (concrete implementation)
   └── GNN (inference events)
   └── Your Custom Event Class

Core Properties
---------------

Event Identification
~~~~~~~~~~~~~~~~~~~~

.. py:attribute:: index
   
   Event index/number identifier from ROOT file.
   
   :type: int

   **Usage**:
   
   .. code-block:: python
   
      event.index = 12345  # Set event number
      print(f"Event: {event.index}")

.. py:attribute:: hash
   
   Unique hash identifier for this event instance.
   
   :type: str

   The hash is computed from event data and used for event comparison and serialization.

.. py:attribute:: Name
   
   Name/identifier for this event type.
   
   :type: str

Event Weighting
~~~~~~~~~~~~~~~

.. py:attribute:: weight
   
   Statistical weight applied to this event.
   
   :type: float
   
   **Purpose**: Used for cross-section weighting, luminosity scaling, pile-up reweighting, etc.
   
   **Example**:
   
   .. code-block:: python
   
      # Apply cross-section weight
      event.weight = xsec * lumi / n_events
      
      # Or reference a ROOT branch
      event.weight = "eventWeight"

ROOT Integration
~~~~~~~~~~~~~~~~

.. py:attribute:: Tree
   
   Name of the ROOT TTree currently being read.
   
   :type: str

.. py:attribute:: Trees
   
   List of ROOT TTree names to process.
   
   :type: list[str]
   
   **Example**:
   
   .. code-block:: python
   
      # Single tree
      event.Trees = "nominal"
      
      # Multiple trees for systematics
      event.Trees = ["nominal", "JET_JER_UP", "JET_JER_DOWN"]

.. py:attribute:: Branches
   
   List of ROOT branch names to read for this event.
   
   :type: list[str]
   
   **Purpose**: Specify which branches to load from ROOT file to optimize I/O.
   
   **Example**:
   
   .. code-block:: python
   
      event.Branches = ["jet_*", "el_*", "mu_*", "met_*"]

Particle Collections
~~~~~~~~~~~~~~~~~~~~

.. py:attribute:: Particles
   
   Collection of all particles in this event.
   
   :type: list[ParticleTemplate]
   
   **Note**: Particles are automatically populated when reading ROOT files.
   
   **Example**:
   
   .. code-block:: python
   
      # Access particles
      for particle in event.Particles:
          if particle.pt > 25e3:  # pT in MeV
              print(f"High-pT particle: pt={particle.pt/1e3:.1f} GeV")
      
      # Filter particles
      jets = [p for p in event.Particles if p.Type == "jet"]
      leptons = [p for p in event.Particles if p.is_lep]

Methods to Override
-------------------

selection()
~~~~~~~~~~~

.. py:method:: selection() -> bool
   
   Define event selection criteria.
   
   :returns: True if event passes selection, False otherwise
   :rtype: bool
   
   **Purpose**: This is the PRIMARY method you should override in your custom event class.
   It determines whether an event passes your analysis cuts.
   
   **Called by**: The Analysis framework automatically calls this method for each event.
   
   **Implementation Guidelines**:
   
   * Return True for events that pass all selection requirements
   * Return False for events that fail any cut
   * Keep selection logic simple and readable
   * Use early returns for failed cuts to optimize performance
   * Do NOT modify event data in this method
   
   **Example Implementation**:
   
   .. code-block:: python
   
      from AnalysisG.core import EventTemplate
      
      class MyEvent(EventTemplate):
          def __init__(self):
              super().__init__()
          
          def selection(self):
              # Get particle collections
              electrons = [p for p in self.Particles 
                          if abs(p.pdgid) == 11 and p.pt > 25e3]
              muons = [p for p in self.Particles 
                      if abs(p.pdgid) == 13 and p.pt > 25e3]
              jets = [p for p in self.Particles 
                     if p.Type == "jet" and p.pt > 30e3 and abs(p.eta) < 2.5]
              
              # Apply cuts
              leptons = electrons + muons
              if len(leptons) < 2:
                  return False  # Require at least 2 leptons
              
              if len(jets) < 4:
                  return False  # Require at least 4 jets
              
              # Check for b-jets
              bjets = [j for j in jets if j.is_b]
              if len(bjets) < 2:
                  return False  # Require at least 2 b-jets
              
              # All cuts passed
              return True

strategy()
~~~~~~~~~~

.. py:method:: strategy()
   
   Define custom event processing strategy.
   
   :returns: None
   
   **Purpose**: Perform event-level calculations, reconstructions, or custom processing
   on events that pass selection.
   
   **Called by**: The Analysis framework calls this AFTER selection() returns True.
   
   **Use Cases**:
   
   * Reconstruct composite particles (W bosons, top quarks, etc.)
   * Calculate event-level observables
   * Apply corrections or calibrations
   * Build object associations
   
   **Example Implementation**:
   
   .. code-block:: python
   
      def strategy(self):
          # Get particles
          leptons = [p for p in self.Particles if p.is_lep]
          jets = [p for p in self.Particles if p.Type == "jet"]
          bjets = [j for j in jets if j.is_b]
          
          # Reconstruct W bosons from lepton pairs
          self.reconstructed_W = []
          if len(leptons) >= 2:
              for i, l1 in enumerate(leptons):
                  for l2 in leptons[i+1:]:
                      if l1.charge * l2.charge < 0:  # Opposite sign
                          W_candidate = l1 + l2
                          if 60e3 < W_candidate.mass < 100e3:  # W mass window
                              self.reconstructed_W.append(W_candidate)
          
          # Reconstruct top quarks from W + b-jet
          self.reconstructed_tops = []
          for W in self.reconstructed_W:
              for b in bjets:
                  top_candidate = W + b
                  if 150e3 < top_candidate.mass < 200e3:  # Top mass window
                      self.reconstructed_tops.append(top_candidate)
          
          # Store number of reconstructed objects
          self.n_W = len(self.reconstructed_W)
          self.n_tops = len(self.reconstructed_tops)

Complete Usage Example
----------------------

Here's a complete example showing how to create a custom event class for a 4-top analysis:

.. code-block:: python

   from AnalysisG.core import EventTemplate, ParticleTemplate
   
   class FourTopEvent(EventTemplate):
       """Event class for 4-top quark analysis."""
       
       def __init__(self):
           super().__init__()
           # Initialize custom properties
           self.reconstructed_tops = []
           self.n_good_jets = 0
           self.n_bjets = 0
       
       def selection(self):
           """Select events suitable for 4-top analysis."""
           # Get particle collections with kinematic cuts
           electrons = [p for p in self.Particles 
                       if abs(p.pdgid) == 11 
                       and p.pt > 25e3  # 25 GeV
                       and abs(p.eta) < 2.47]
           
           muons = [p for p in self.Particles 
                   if abs(p.pdgid) == 13 
                   and p.pt > 25e3  # 25 GeV
                   and abs(p.eta) < 2.5]
           
           jets = [p for p in self.Particles 
                  if p.Type == "jet" 
                  and p.pt > 30e3  # 30 GeV
                  and abs(p.eta) < 2.5]
           
           # Lepton selection: exactly 2 opposite-sign same-flavor leptons
           leptons = electrons + muons
           if len(leptons) != 2:
               return False
           
           if leptons[0].charge * leptons[1].charge >= 0:
               return False  # Must be opposite sign
           
           # Jet multiplicity
           self.n_good_jets = len(jets)
           if self.n_good_jets < 6:
               return False  # Need at least 6 jets for 4-top
           
           # B-jet requirement
           bjets = [j for j in jets if j.is_b]
           self.n_bjets = len(bjets)
           if self.n_bjets < 3:
               return False  # Need at least 3 b-jets
           
           # All cuts passed
           return True
       
       def strategy(self):
           """Reconstruct top quarks from selected events."""
           # Get particles (we know they pass selection now)
           leptons = [p for p in self.Particles if p.is_lep and p.pt > 25e3]
           jets = [p for p in self.Particles if p.Type == "jet" and p.pt > 30e3]
           bjets = [j for j in jets if j.is_b]
           
           # Simple top reconstruction: lepton + b-jet combinations
           self.reconstructed_tops = []
           for lepton in leptons:
               for bjet in bjets:
                   # Require minimum angular separation
                   if lepton.DeltaR(bjet) < 0.4:
                       continue
                   
                   # Combine lepton and b-jet (missing neutrino approximation)
                   top_candidate = lepton + bjet
                   
                   # Loose top mass requirement
                   if 120e3 < top_candidate.mass < 220e3:
                       self.reconstructed_tops.append(top_candidate)

   # Usage in analysis
   from AnalysisG.core import Analysis
   
   analysis = Analysis()
   analysis.AddEvent(FourTopEvent(), "4top")
   analysis.AddSamples("/path/to/data/*.root", "4top_signal")
   analysis.OutputPath = "./output"
   analysis.Start()

Best Practices
--------------

1. **Always call super().__init__()** in your constructor
2. **Keep selection() pure**: Don't modify event state, only check conditions
3. **Use strategy() for modifications**: Reconstruction and calculations go here
4. **Cache expensive calculations**: Store results as instance attributes
5. **Handle edge cases**: Check for empty lists before accessing indices
6. **Use meaningful variable names**: Make your code self-documenting
7. **Units matter**: ROOT typically uses MeV, be explicit in comparisons
8. **Test incrementally**: Start with simple selections and add complexity

See Also
--------

* :doc:`../interfaces/overview`: Overview of simple interfaces
* :doc:`particle_template`: ParticleTemplate documentation
* :doc:`../events/overview`: Concrete event implementations
* :doc:`analysis`: Analysis orchestration class
