Core Module
===========

The core module provides the fundamental classes and utilities for AnalysisG.

Overview
--------

The core module is implemented in Cython and provides Python bindings to the C++ backend. 
All core classes are located in ``src/AnalysisG/core/`` and include both ``.pyx`` (Cython implementation) 
and ``.pxd`` (Cython header) files.

Main Components
---------------

Analysis
~~~~~~~~

.. autoclass:: AnalysisG.core.analysis.Analysis
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The Analysis class is the main entry point for running analyses. It orchestrates:

* Sample loading and processing
* Event and particle template registration
* Graph construction for GNN applications
* Selection criteria application
* Model training and inference
* Metric evaluation

Key Methods
^^^^^^^^^^^

* ``AddSamples(path, label)``: Add ROOT samples to analyze
* ``AddEvent(event_template, label)``: Register an event template
* ``AddGraph(graph_template, label)``: Register a graph template
* ``AddSelection(selection_template)``: Register a selection template
* ``AddMetric(metric_template, model_template)``: Register evaluation metrics
* ``AddModel(model_template, optimizer_config, run_name)``: Add a model for training
* ``AddModelInference(model_template, run_name)``: Add a model for inference only
* ``Start()``: Begin the analysis execution

Properties
^^^^^^^^^^

* ``OutputPath``: Directory for output files
* ``FetchMeta``: Whether to fetch metadata from PyAMI
* ``PreTagEvents``: Whether to pre-tag events for efficiency

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from AnalysisG import Analysis
   from AnalysisG.core.event_template import EventTemplate
   
   # Create analysis instance
   ana = Analysis()
   
   # Configure analysis
   ana.AddSamples("/path/to/samples", "signal")
   ana.AddEvent(MyEventTemplate(), "my_events")
   
   # Run analysis
   ana.Start()

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

The Analysis class wraps a C++ ``analysis`` object that handles the core processing logic. 
The Cython implementation manages memory through ``__cinit__``, ``__init__``, and ``__dealloc__`` 
methods to ensure proper resource management.

C++ Backend
^^^^^^^^^^^

The C++ implementation provides:

* Efficient event processing
* Multi-threaded sample loading
* Memory-efficient data structures
* Integration with ROOT I/O

See Also
--------

* :doc:`core/templates` - Template classes for events, particles, graphs, etc.
* :doc:`core/io` - ROOT I/O functionality
* :doc:`core/tools` - Utility functions
