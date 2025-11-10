Modules
=======

The ``modules`` directory contains core framework functionality that powers AnalysisG.

Overview
--------

This module provides essential building blocks including:

- **Analysis**: Main analysis compiler for defining action chains
- **Container**: Data container classes for managing physics objects
- **DataLoader**: Efficient data loading and batching
- **Event**: Event template system
- **Graph**: Graph template for neural networks
- **I/O**: ROOT file interface
- **Loss Functions**: Training loss implementations
- **Meta**: Metadata and introspection utilities
- **Metrics**: Performance metrics
- **Model**: Machine learning model templates
- **Notification**: User notification system
- **Optimizer**: Training optimizers
- **Particle**: Particle template system
- **Plotting**: Visualization utilities
- **SampleTracer**: Data provenance tracking
- **Selection**: Event selection framework
- **Structs**: Core data structures
- **Tools**: Utility functions
- **TypeCasting**: Type conversion utilities

Key Template Classes
--------------------

The modules directory provides the following key template classes that users inherit from:

Analysis Module
~~~~~~~~~~~~~~~

The ``analysis`` class is the main entry point for defining analysis workflows. It orchestrates:

- Event loading from ROOT files
- Applying selections
- Training models
- Generating outputs

Event Template
~~~~~~~~~~~~~~

The ``event_template`` base class defines the interface for physics events:

- Property system for event attributes
- Tree/branch/leaf mapping to ROOT
- Virtual methods: ``clone()``, ``build()``, ``CompileEvent()``

Particle Template
~~~~~~~~~~~~~~~~~

The ``particle_template`` represents individual physics objects:

- 4-momentum representation  
- Parent-child relationships for decay chains
- Truth matching capabilities

Graph Template
~~~~~~~~~~~~~~

The ``graph_template`` is used for GNN applications:

- Node features from particles
- Edge features from particle pairs
- Global graph features from events

Model Template
~~~~~~~~~~~~~~

The ``model_template`` standardizes ML models:

- Training/validation loops
- Optimizer and loss configuration
- Metric tracking and logging

Selection Template
~~~~~~~~~~~~~~~~~~

The ``selection_template`` enables cut-based analyses:

- Event selection logic
- Histogram management
- Cutflow tracking

API Reference
-------------

The complete API documentation for module classes is automatically generated from source code using Doxygen and displayed below.

Analysis Class
~~~~~~~~~~~~~~

.. doxygenclass:: analysis
   :members:

Container Class
~~~~~~~~~~~~~~~

.. doxygenclass:: container
   :members:

DataLoader Class
~~~~~~~~~~~~~~~~

.. doxygenclass:: dataloader
   :members:
