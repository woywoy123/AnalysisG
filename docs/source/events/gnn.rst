GNN Event
=========

GNN inference event implementation.

File Location
~~~~~~~~~~~~~

* **Event Implementation**: ``src/AnalysisG/events/gnn/event_gnn.pyx``
* **Particle Implementation**: ``src/AnalysisG/events/gnn/particle_gnn.pyx``
* **C++ Event Header**: ``src/AnalysisG/events/gnn/include/gnn/event.h``
* **C++ Particle Header**: ``src/AnalysisG/events/gnn/include/gnn/particles.h``

Description
-----------

This event type is designed for specific physics analyses.

Components
----------

* Event class: Inherits from EventTemplate
* Particle class: Inherits from ParticleTemplate
* C++ backend: Efficient data handling

See Also
--------

* :doc:`overview`: Events package overview
* :doc:`../interfaces/event_interface`: Event interface
