Simple Interfaces Overview
===========================

The AnalysisG framework provides a set of simple, user-friendly interfaces that are designed to be extended and customized for your specific analysis needs. These interfaces are high-level template classes that hide the complexity of the underlying C++/CUDA implementations.

What are Simple Interfaces?
----------------------------

Simple interfaces are Python/Cython classes that:

* Provide clear, well-defined APIs
* Are designed to be inherited and customized
* Have methods that users are expected to override
* Abstract away low-level implementation details
* Focus on physics analysis rather than technical implementation

Core Template Classes
----------------------

The following template classes form the foundation of the simple interfaces:

EventTemplate
~~~~~~~~~~~~~
Base class for defining custom event structures. Override this to implement event-level selection, processing, and analysis logic.

**Key Methods to Override:**

* ``selection()``: Define event selection criteria
* ``apply_weight()``: Apply event weights
* ``process()``: Custom event processing

ParticleTemplate
~~~~~~~~~~~~~~~~
Base class for defining custom particle types. Override this to implement particle-level operations and properties.

**Key Methods to Override:**

* ``is_lepton()``, ``is_jet()``, etc.: Particle type identification
* ``compute_features()``: Calculate custom particle features
* ``apply_corrections()``: Apply detector corrections

GraphTemplate
~~~~~~~~~~~~~
Base class for defining graph representations of events. Override this to implement custom graph construction logic.

**Key Methods to Override:**

* ``build_graph()``: Construct the graph structure
* ``add_edges()``: Define edge connectivity
* ``compute_edge_features()``: Calculate edge features

MetricTemplate
~~~~~~~~~~~~~~
Base class for defining custom metrics. Override this to implement evaluation metrics for your analysis.

**Key Methods to Override:**

* ``compute()``: Calculate the metric value
* ``update()``: Update metric with new predictions
* ``reset()``: Reset metric state

ModelTemplate
~~~~~~~~~~~~~
Base class for defining machine learning models. Override this to implement custom model architectures.

**Key Methods to Override:**

* ``forward()``: Model forward pass
* ``loss()``: Loss function computation
* ``predict()``: Generate predictions

SelectionTemplate
~~~~~~~~~~~~~~~~~
Base class for defining selection criteria. Override this to implement custom selection logic.

**Key Methods to Override:**

* ``apply()``: Apply selection to events
* ``passes()``: Check if object passes selection

When to Use Simple Interfaces
------------------------------

Use simple interfaces when:

* Implementing physics analyses
* Defining custom event/particle types
* Creating new selection criteria
* Developing analysis-specific algorithms
* You need a high-level, Python-friendly API

For low-level optimization and performance-critical code, see the :doc:`../technical/overview` section.

Example Usage
-------------

Here's a simple example of using the interfaces:

.. code-block:: python

   from AnalysisG.core import EventTemplate, ParticleTemplate
   
   class MyParticle(ParticleTemplate):
       def is_lepton(self):
           return abs(self.pdgid) in [11, 13, 15]
   
   class MyEvent(EventTemplate):
       def selection(self):
           # Select events with at least 2 leptons
           leptons = [p for p in self.Particles if p.is_lepton()]
           return len(leptons) >= 2

See Also
--------

* :doc:`core_templates`: Detailed documentation of core template classes
* :doc:`event_interface`: Event interface details
* :doc:`particle_interface`: Particle interface details
* :doc:`../technical/overview`: Complex technical components
