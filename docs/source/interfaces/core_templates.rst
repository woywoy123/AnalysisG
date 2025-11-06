Core Templates
==============

The core package contains the fundamental template classes that form the basis of the AnalysisG framework. These templates are designed to be inherited and customized by users.

EventTemplate
-------------

.. class:: EventTemplate

   Base template class for event-level data structures.

   The EventTemplate class provides the foundation for defining custom event types. Users should inherit from this class and override methods to implement custom event processing logic.

   **Key Properties**:

   .. property:: index
      
      Event index identifier.
      
      :type: int

   .. property:: weight
      
      Event weight for statistical analysis.
      
      :type: float

   .. property:: Tree
      
      ROOT tree name associated with this event.
      
      :type: str

   .. property:: Trees
      
      List of ROOT tree names.
      
      :type: list[str]

   .. property:: Branches
      
      List of ROOT branches to read.
      
      :type: list[str]

   .. property:: Particles
      
      List of particles in this event.
      
      :type: list[ParticleTemplate]

   **Key Methods**:

   .. method:: selection() -> bool
      
      Event selection criteria. Override this method to implement custom event selection logic.
      
      :returns: True if event passes selection, False otherwise
      :rtype: bool

   .. method:: strategy()
      
      Event processing strategy. Override to define how the event should be analyzed.

   .. method:: merge(EventTemplate other)
      
      Merge another event with this one.
      
      :param other: Event to merge
      :type other: EventTemplate

   **Usage Example**:

   .. code-block:: python

      from AnalysisG.core import EventTemplate

      class MyEvent(EventTemplate):
          def __init__(self):
              super().__init__()
          
          def selection(self):
              # Require at least 4 jets
              jets = [p for p in self.Particles if p.is_jet()]
              return len(jets) >= 4

ParticleTemplate
----------------

.. class:: ParticleTemplate

   Base template class for particle-level data structures.

   The ParticleTemplate class represents individual particles in an event. It provides methods for accessing particle properties and implementing custom particle identification logic.

   **Key Properties**:

   .. property:: px
      
      Particle momentum x-component (GeV).
      
      :type: float

   .. property:: py
      
      Particle momentum y-component (GeV).
      
      :type: float

   .. property:: pz
      
      Particle momentum z-component (GeV).
      
      :type: float

   .. property:: e
      
      Particle energy (GeV).
      
      :type: float

   .. property:: pt
      
      Transverse momentum (GeV).
      
      :type: float

   .. property:: eta
      
      Pseudorapidity.
      
      :type: float

   .. property:: phi
      
      Azimuthal angle (radians).
      
      :type: float

   .. property:: mass
      
      Particle mass (GeV).
      
      :type: float

   .. property:: charge
      
      Electric charge.
      
      :type: float

   .. property:: pdgid
      
      PDG particle ID code.
      
      :type: int

   .. property:: Parents
      
      List of parent particles.
      
      :type: list[ParticleTemplate]

   .. property:: Children
      
      List of child particles.
      
      :type: list[ParticleTemplate]

   **Key Methods**:

   .. method:: is_lepton() -> bool
      
      Check if particle is a lepton.
      
      :returns: True if particle is a lepton
      :rtype: bool

   .. method:: is_jet() -> bool
      
      Check if particle is a jet.
      
      :returns: True if particle is a jet
      :rtype: bool

   .. method:: DeltaR(ParticleTemplate other) -> float
      
      Calculate angular separation (ΔR) from another particle.
      
      :param other: Other particle
      :type other: ParticleTemplate
      :returns: Angular separation
      :rtype: float

   .. method:: clone() -> ParticleTemplate
      
      Create a copy of this particle.
      
      :returns: Cloned particle
      :rtype: ParticleTemplate

   **Operator Overloads**:

   The ParticleTemplate class supports arithmetic operations for four-momentum addition:

   .. code-block:: python

      # Add two particles' four-momenta
      combined = particle1 + particle2
      
      # In-place addition
      particle1 += particle2

GraphTemplate
-------------

.. class:: GraphTemplate

   Base template class for graph representations of events.

   The GraphTemplate class provides functionality for creating graph-based representations of physics events, suitable for Graph Neural Network analyses.

   **Key Properties**:

   .. property:: Nodes
      
      List of graph nodes (particles).
      
      :type: list

   .. property:: Edges
      
      List of graph edges (connections between particles).
      
      :type: list

   .. property:: NodeFeatures
      
      Node feature matrix.
      
      :type: tensor

   .. property:: EdgeFeatures
      
      Edge feature matrix.
      
      :type: tensor

   **Key Methods**:

   .. method:: build_graph()
      
      Construct the graph structure. Override this method to implement custom graph building logic.

   .. method:: add_node(ParticleTemplate particle)
      
      Add a node to the graph.
      
      :param particle: Particle to add as a node
      :type particle: ParticleTemplate

   .. method:: add_edge(int source, int target)
      
      Add an edge between two nodes.
      
      :param source: Source node index
      :type source: int
      :param target: Target node index
      :type target: int

See Also
--------

* :doc:`../interfaces/overview`: Overview of simple interfaces
* :doc:`event_template`: Detailed EventTemplate documentation
* :doc:`particle_template`: Detailed ParticleTemplate documentation
* :doc:`graph_template`: Detailed GraphTemplate documentation
