Templates
=========

Template classes that define the core interfaces for AnalysisG components.

Overview
--------

The ``templates`` module provides abstract base classes that users extend to create custom:

- Event definitions
- Particle types  
- Graph structures
- Machine learning models
- Performance metrics
- Selection algorithms

These templates ensure consistency across the framework and enable polymorphic behavior.

Template Hierarchy
------------------

Event Template System
~~~~~~~~~~~~~~~~~~~~~

The ``event_template`` is the central abstraction for physics events. It provides:

- Property system for event attributes
- Tree/branch/leaf mapping for ROOT I/O
- Abstract methods for building and compiling events
- Integration with the analysis framework

Key methods:
- ``clone()``: Create event copies
- ``build(element_t* el)``: Populate from ROOT
- ``CompileEvent()``: Finalize after building

Particle Template System
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``particle_template`` represents individual physics objects (jets, leptons, etc.). Features:

- 4-momentum representation
- Parent-child relationships for decay chains
- Truth matching capabilities
- Customizable properties

Attributes:
- ``pt``, ``eta``, ``phi``, ``e``: Kinematics
- ``Parents``, ``Children``: Decay chain
- ``index``: Unique identifier

Graph Template System
~~~~~~~~~~~~~~~~~~~~~

For graph neural network applications, the ``graph_template`` defines:

- Node features from particles
- Edge features from particle pairs
- Global graph features from events
- Automatic feature tensor construction

Methods:
- ``NodeFeature()``: Extract node features
- ``EdgeFeature()``: Compute edge features
- ``GraphFeature()``: Extract graph-level features

Model Template System
~~~~~~~~~~~~~~~~~~~~~

The ``model_template`` standardizes machine learning models:

- Training/validation/test loops
- Optimizer configuration
- Loss function specification
- Metric tracking
- Checkpointing

Workflow:
1. Define model architecture
2. Specify loss and metrics
3. Configure optimizer
4. Train with automatic logging

Selection Template System
~~~~~~~~~~~~~~~~~~~~~~~~~~

For cut-based analyses, the ``selection_template`` provides:

- Event selection logic
- Histogram booking
- Cutflow tracking
- Result serialization

Methods:
- ``Selection()``: Define cuts
- ``InitHistograms()``: Book histograms
- ``ApplySelection()``: Process events
- ``Finalize()``: Generate output

Metric Template System
~~~~~~~~~~~~~~~~~~~~~~

Custom performance metrics via the ``metric_template``:

- Accumulation across batches
- Reduction strategies
- Metric computation
- Logging integration

Interface:
- ``Accumulate()``: Update with batch
- ``Compute()``: Calculate final value
- ``Reset()``: Clear for new epoch

Design Philosophy
-----------------

The template system follows several key principles:

1. **Polymorphism**: All templates use virtual methods for extensibility
2. **Composition**: Templates can be combined and nested
3. **Type Safety**: Cython provides compile-time type checking
4. **Minimal Overhead**: Templates add minimal runtime cost
5. **Flexibility**: Users can override any behavior

Template Usage Pattern
----------------------

The typical workflow for using templates:

1. **Inherit**: Create a class inheriting from the appropriate template
2. **Implement**: Override abstract/virtual methods with custom logic
3. **Register**: Register your class with the analysis framework
4. **Execute**: The framework calls your implementations polymorphically

This pattern enables:

- Code reuse across analyses
- Standardized interfaces
- Easy testing and validation
- Interoperability between components

Interrelationships
------------------

Templates work together to form complete workflows:

- **Events** contain **Particles**
- **Graphs** are built from **Events** and **Particles**
- **Models** train on **Graphs** using **Metrics**
- **Selections** filter **Events** and create histograms
- **Analysis** orchestrates all components

This modular design allows mixing and matching components for different analyses.

API Reference
-------------

The complete API documentation for template classes is generated from source code using Doxygen.

.. note::
   To view the full API reference with all template classes, build the Doxygen documentation locally:
   
   .. code-block:: bash
   
      cd docs
      doxygen Doxyfile
      # Open build/doxygen/html/index.html

Key Template Classes
~~~~~~~~~~~~~~~~~~~~

The following template classes form the core of the AnalysisG framework:

- ``event_template`` - Base class for event definitions
- ``particle_template`` - Base class for particle types
- ``graph_template`` - Base class for graph structures
- ``model_template`` - Base class for ML models
- ``selection_template`` - Base class for event selections
- ``metric_template`` - Base class for performance metrics
