Module Documentation
====================

This section provides detailed documentation for each module in the AnalysisG
framework, organized by functionality.

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   core
   events
   graphs
   metrics
   models
   infrastructure
   python_interface
   templates

Overview
--------

The AnalysisG framework is organized into several key modules, each providing
specific functionality for high-energy physics analysis with graph neural networks.

Core Modules
------------

* :doc:`core` - Fundamental building blocks and base templates
* :doc:`events` - Event processing for various physics analyses
* :doc:`graphs` - Graph construction from physics events
* :doc:`metrics` - Performance metrics and evaluation
* :doc:`models` - Neural network model implementations

Infrastructure
--------------

* :doc:`infrastructure` - Low-level infrastructure components
* :doc:`python_interface` - Python-C++ interface via Cython
* :doc:`templates` - Code templates and examples

Module Relationships
--------------------

The modules have the following dependency structure:

.. code-block:: text

   Core (templates)
   ├── Events (inherits from event_template)
   ├── Graphs (inherits from graph_template)
   ├── Models (inherits from model_template)
   └── Metrics (inherits from metric_template)
   
   Infrastructure
   ├── Used by all modules
   └── Provides low-level functionality
   
   Python Interface
   └── Wraps all modules for Python access
   
   Templates
   └── Provides examples for all module types

Navigation
----------

* Use the sidebar to navigate between modules
* Each module page includes detailed documentation
* API reference provides low-level details
* Examples demonstrate common usage patterns

For API details, see :doc:`../api/index`.
