Quick Start Guide
=================

This guide will help you get started with AnalysisG quickly.

Basic Usage
-----------

Creating a Custom Event
~~~~~~~~~~~~~~~~~~~~~~~

To create a custom event, inherit from the ``EventTemplate`` class:

.. code-block:: python

   from AnalysisG.core import EventTemplate

   class MyEvent(EventTemplate):
       def __init__(self):
           super().__init__()
       
       # Override methods as needed
       def selection(self):
           # Implement event selection logic
           return True

Creating a Custom Particle
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a custom particle, inherit from the ``ParticleTemplate`` class:

.. code-block:: python

   from AnalysisG.core import ParticleTemplate

   class MyParticle(ParticleTemplate):
       def __init__(self):
           super().__init__()
       
       # Add custom properties and methods

Running an Analysis
~~~~~~~~~~~~~~~~~~~

To run an analysis, use the ``Analysis`` class:

.. code-block:: python

   from AnalysisG.core import Analysis
   
   # Create analysis instance
   analysis = Analysis()
   
   # Configure analysis
   analysis.Event = MyEvent
   analysis.Particle = MyParticle
   
   # Run analysis
   analysis.Run()

Working with Graphs
-------------------

AnalysisG supports graph-based analyses:

.. code-block:: python

   from AnalysisG.core import GraphTemplate
   
   class MyGraph(GraphTemplate):
       def __init__(self):
           super().__init__()
       
       def build_graph(self):
           # Implement graph construction logic
           pass

Using GPU Acceleration
----------------------

To use GPU-accelerated operations:

.. code-block:: python

   from AnalysisG.pyc import operators
   
   # GPU-accelerated operations will be used automatically if CUDA is available
   result = operators.some_operation(data)

Next Steps
----------

* Explore the :doc:`interfaces/core_templates` for available template classes
* Read about :doc:`../events/overview` to understand event structures
* Learn about :doc:`../technical/overview` for advanced usage
