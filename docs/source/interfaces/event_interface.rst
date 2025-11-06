Event Interface
===============

The Event Interface provides the foundation for defining custom event types in AnalysisG.

Overview
--------

Events are the primary data containers in HEP analyses. The EventTemplate class provides a flexible interface for:

* Reading event data from ROOT files
* Implementing event selection criteria
* Processing and transforming event-level information
* Organizing particles within events
* Managing event weights and metadata

Core EventTemplate Class
------------------------

File Location
~~~~~~~~~~~~~

* **Cython Implementation**: ``src/AnalysisG/core/event_template.pyx``
* **Cython Header**: ``src/AnalysisG/core/event_template.pxd``

Class Definition
~~~~~~~~~~~~~~~~

.. class:: EventTemplate

   The base class for all event types in AnalysisG.

Properties
~~~~~~~~~~

Event Identification
^^^^^^^^^^^^^^^^^^^^

.. property:: index
   
   Unique identifier for the event.
   
   :type: int

.. property:: hash
   
   Hash value for event identification and comparison.
   
   :type: str

Event Data
^^^^^^^^^^

.. property:: Particles
   
   Collection of particles in this event.
   
   :type: list[ParticleTemplate]
   :getter: Returns list of particles
   :setter: Assigns particles to event

.. property:: weight
   
   Statistical weight for this event.
   
   :type: float
   :getter: Returns event weight
   :setter: Sets event weight

ROOT Integration
^^^^^^^^^^^^^^^^

.. property:: Tree
   
   Name of the ROOT TTree containing this event.
   
   :type: str
   :getter: Returns tree name
   :setter: Sets tree name

.. property:: Trees
   
   List of all ROOT TTrees to process.
   
   :type: list[str]
   :getter: Returns list of tree names
   :setter: Sets tree names

.. property:: Branches
   
   List of ROOT branches to read from the tree.
   
   :type: list[str]
   :getter: Returns branch names
   :setter: Sets branch names to read

Methods to Override
~~~~~~~~~~~~~~~~~~~

Selection Method
^^^^^^^^^^^^^^^^

.. method:: selection() -> bool
   
   Define event selection criteria.
   
   This method should return True if the event passes your analysis selection requirements,
   and False otherwise. This is the primary method users should override.
   
   :returns: Whether the event passes selection
   :rtype: bool
   
   **Example**:
   
   .. code-block:: python
   
      def selection(self):
          # Require at least 2 leptons and 4 jets
          leptons = [p for p in self.Particles if p.is_lep]
          jets = [p for p in self.Particles if p.Type == "jet"]
          return len(leptons) >= 2 and len(jets) >= 4

Strategy Method
^^^^^^^^^^^^^^^

.. method:: strategy()
   
   Define custom event processing strategy.
   
   Override this method to implement custom event-level calculations, reconstructions,
   or other processing that should be performed on selected events.
   
   **Example**:
   
   .. code-block:: python
   
      def strategy(self):
          # Reconstruct top quarks from particles
          self.reconstruct_tops()
          self.calculate_observables()

Utility Methods
~~~~~~~~~~~~~~~

Comparison and Hashing
^^^^^^^^^^^^^^^^^^^^^^^

.. method:: __hash__() -> int
   
   Return hash value for this event.

.. method:: __eq__(EventTemplate other) -> bool
   
   Compare two events for equality.
   
   :param other: Event to compare with
   :type other: EventTemplate
   :returns: True if events are equal
   :rtype: bool

Merging
^^^^^^^

.. method:: merge(EventTemplate other)
   
   Merge data from another event.
   
   :param other: Event to merge from
   :type: EventTemplate

Type Checking
^^^^^^^^^^^^^

.. method:: is_self(inpt) -> bool
   
   Check if input is an EventTemplate instance.
   
   :param inpt: Object to check
   :returns: True if input is EventTemplate or subclass
   :rtype: bool

Usage Examples
--------------

Basic Event Selection
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import EventTemplate
   
   class MyEvent(EventTemplate):
       def __init__(self):
           super().__init__()
       
       def selection(self):
           # Select events with high jet multiplicity
           jets = [p for p in self.Particles if p.Type == "jet"]
           return len(jets) >= 6

Event with Custom Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import EventTemplate
   
   class TopEvent(EventTemplate):
       def __init__(self):
           super().__init__()
           self.reconstructed_tops = []
       
       def selection(self):
           # Basic selection
           return len(self.Particles) > 0
       
       def strategy(self):
           # Custom reconstruction
           jets = [p for p in self.Particles if p.Type == "jet"]
           leptons = [p for p in self.Particles if p.is_lep]
           
           # Reconstruct top candidates
           for l in leptons:
               for j1 in jets:
                   for j2 in jets:
                       if j1 != j2:
                           top_candidate = l + j1 + j2
                           self.reconstructed_tops.append(top_candidate)

Event with Weights
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import EventTemplate
   
   class WeightedEvent(EventTemplate):
       def __init__(self):
           super().__init__()
       
       def selection(self):
           # Apply event weight based on conditions
           if self.some_condition():
               self.weight = 1.5
           else:
               self.weight = 1.0
           return True

Best Practices
--------------

1. **Always call super().__init__()**: Initialize the parent class in your constructor.

2. **Keep selection simple**: The selection() method should return a boolean. Complex logic should go in strategy().

3. **Use properties**: Access particle collections and event metadata through properties, not direct attribute access.

4. **Cache results**: If you perform expensive calculations in strategy(), store the results as instance attributes.

5. **Handle edge cases**: Check for empty particle lists or missing data before processing.

See Also
--------

* :doc:`particle_interface`: Particle interface documentation
* :doc:`../core/event_template`: Core EventTemplate implementation details
* :doc:`../events/overview`: Concrete event implementations
