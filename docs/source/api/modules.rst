Infrastructure Modules
======================

The Modules directory contains fundamental infrastructure components.

For complete API reference, see the Doxygen-generated HTML documentation in ``doxygen-docs/html/``.

Module Components
-----------------

Templates
~~~~~~~~~

Base template classes defining interfaces:

* ``event_template.h`` - Event processing
* ``particle_template.h`` - Particle representation
* ``graph_template.h`` - Graph construction
* ``model_template.h`` - Neural network models
* ``metric_template.h`` - Performance metrics
* ``selection_template.h`` - Event selection

Container
~~~~~~~~~

Data container utilities for efficient storage.

**Location**: ``src/AnalysisG/modules/container/``

DataLoader
~~~~~~~~~~

Data loading and batching infrastructure.

**Location**: ``src/AnalysisG/modules/dataloader/``

Features:
* ROOT file reading
* Batch preparation
* Multi-threaded pipeline
* Caching and prefetching

IO Module
~~~~~~~~~

Input/output operations.

**Location**: ``src/AnalysisG/modules/io/``

Graph Module
~~~~~~~~~~~~

Graph utilities and operations.

**Location**: ``src/AnalysisG/modules/graph/``

Meta Module
~~~~~~~~~~~

Metadata management and tracking.

**Location**: ``src/AnalysisG/modules/meta/``

Structs Module
~~~~~~~~~~~~~~

Common data structures.

**Location**: ``src/AnalysisG/modules/structs/``

Key structures:
* ``property.h`` - Property system
* ``element.h`` - Element definitions
* ``event.h`` - Event structures

Tools Module
~~~~~~~~~~~~

Utility functions and tools.

**Location**: ``src/AnalysisG/modules/tools/``

Neutrino Solver
~~~~~~~~~~~~~~~

Neutrino reconstruction algorithms.

**Location**: ``src/AnalysisG/modules/nusol/``

Submodules:

* **nusol** - Neutrino solutions
* **conuix** - Constrained solutions
* **ellipse** - Elliptic constraint methods
* **multisol** - Multi-solution handling

Features:
* Analytic reconstruction
* Kinematic constraints
* Multiple solution handling
* CUDA acceleration

Optimizer
~~~~~~~~~

Training optimization utilities.

**Location**: ``src/AnalysisG/modules/optimizer/``

Loss Functions
~~~~~~~~~~~~~~

Loss function implementations.

**Location**: ``src/AnalysisG/modules/lossfx/``

Plotting
~~~~~~~~

Visualization and plotting utilities.

**Location**: ``src/AnalysisG/modules/plotting/``

ROC Analysis
~~~~~~~~~~~~

ROC curve analysis tools.

**Location**: ``src/AnalysisG/modules/roc/``

Analysis
~~~~~~~~

Analysis tools and utilities.

**Location**: ``src/AnalysisG/modules/analysis/``

Notification
~~~~~~~~~~~~

Notification and messaging system.

**Location**: ``src/AnalysisG/modules/notification/``

Sample Tracer
~~~~~~~~~~~~~

Sample tracking and management.

**Location**: ``src/AnalysisG/modules/sampletracer/``

Type Casting
~~~~~~~~~~~~

Type conversion utilities.

**Location**: ``src/AnalysisG/modules/typecasting/``

Particle
~~~~~~~~

Particle utilities and operations.

**Location**: ``src/AnalysisG/modules/particle/``

Selection
~~~~~~~~~

Event selection infrastructure.

**Location**: ``src/AnalysisG/modules/selection/``

Model Infrastructure
~~~~~~~~~~~~~~~~~~~~

Model management and checkpointing.

**Location**: ``src/AnalysisG/modules/model/``

Metric Infrastructure
~~~~~~~~~~~~~~~~~~~~~

Metrics collection system.

**Location**: ``src/AnalysisG/modules/metric/``

Design Principles
-----------------

Infrastructure modules follow:

* **Modularity**: Independent, reusable components
* **Performance**: Optimized C++ implementations
* **Flexibility**: Extensible interfaces
* **Integration**: Seamless component interaction
