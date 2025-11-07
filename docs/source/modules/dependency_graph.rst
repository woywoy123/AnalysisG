Complete Module Dependency Analysis
====================================

This document provides a complete recursive dependency analysis of all modules in the AnalysisG framework,
identifying which modules depend on which others and organizing them into logical categories.

Overview
--------

**Total Source Files Analyzed**: 339 files

* 124 header files (.h, .cuh)
* 215 source files (.cxx, .cu, .pyx)

**Module Organization**: 52 distinct modules across 11 categories

Module Categories
-----------------

Core Templates (6 modules)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the fundamental template classes that users extend to implement their analyses.

**event** (3 files)
    * Location: ``src/AnalysisG/modules/event``
    * Dependencies: meta, reconstruction, structs, templates, tools
    * Purpose: Event template for defining event-level selection logic
    * User extends: ``EventTemplate`` class

**particle** (6 files)
    * Location: ``src/AnalysisG/modules/particle``
    * Dependencies: structs, templates, tools
    * Purpose: Particle template for physics object representation
    * User extends: ``ParticleTemplate`` class

**graph** (4 files)
    * Location: ``src/AnalysisG/modules/graph``
    * Dependencies: structs, templates, tools
    * Purpose: Graph template for GNN inputs
    * User extends: ``GraphTemplate`` class

**selection** (4 files)
    * Location: ``src/AnalysisG/modules/selection``
    * Dependencies: meta, structs, templates, tools
    * Purpose: Selection template for physics object filtering
    * User extends: ``SelectionTemplate`` class

**metric** (6 files)
    * Location: ``src/AnalysisG/modules/metric``
    * Dependencies: meta, notification, plotting, structs, templates, tools
    * Purpose: Metric template for evaluation metrics
    * User extends: ``MetricTemplate`` class

**model** (5 files)
    * Location: ``src/AnalysisG/modules/model``
    * Dependencies: notification, structs, templates
    * Purpose: Model template for PyTorch models
    * User extends: ``ModelTemplate`` class

Analysis Infrastructure (4 modules)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These modules orchestrate the analysis workflow and provide training infrastructure.

**analysis** (9 files)
    * Location: ``src/AnalysisG/modules/analysis``
    * Dependencies: AnalysisG, generators, io, structs, templates, tools
    * Purpose: Main analysis orchestrator
    * Key Classes: ``Analysis``
    * Coordinates: Event processing, selection application, graph building, model training

**lossfx** (5 files)
    * Location: ``src/AnalysisG/modules/lossfx``
    * Dependencies: notification, structs, templates, tools
    * Purpose: Loss function and optimizer management
    * Provides: 20 loss functions, 6 optimizers, 2 LR schedulers

**optimizer** (2 files)
    * Location: ``src/AnalysisG/modules/optimizer``
    * Dependencies: generators, metrics, structs, templates
    * Purpose: Training optimization logic

**meta** (2 files)
    * Location: ``src/AnalysisG/modules/meta``
    * Dependencies: notification, structs, tools
    * Purpose: Metadata management and compilation

Data Management (4 modules)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These modules handle file I/O, data loading, and sample management.

**io** (5 files)
    * Location: ``src/AnalysisG/modules/io``
    * Dependencies: meta, notification, structs, tools
    * Purpose: ROOT and HDF5 file I/O
    * Key Classes: ``root_reader``, ``tree_reader``, ``branch_reader``

**container** (3 files)
    * Location: ``src/AnalysisG/modules/container``
    * Dependencies: generators, meta, templates, tools
    * Purpose: Event/graph/selection collections

**dataloader** (4 files)
    * Location: ``src/AnalysisG/modules/dataloader``
    * Dependencies: generators, io, notification, structs, templates, tools
    * Purpose: Batch loading for training
    * Features: CUDA memory management, multi-threading

**sampletracer** (2 files)
    * Location: ``src/AnalysisG/modules/sampletracer``
    * Dependencies: container, generators, notification
    * Purpose: Sample metadata tracking

Utilities (4 modules)
~~~~~~~~~~~~~~~~~~~~~

Core utility modules providing fundamental functionality.

**tools** (4 files)
    * Location: ``src/AnalysisG/modules/tools``
    * Dependencies: (none - foundation module)
    * Purpose: Base utility functions and classes
    * Note: This is a foundation module with no dependencies

**structs** (18 files)
    * Location: ``src/AnalysisG/modules/structs``
    * Dependencies: tools
    * Purpose: Core data structures
    * Defines: ``particle_t``, ``event_t``, ``graph_t``, ``settings_t``, ``property_t``, ``element_t``

**typecasting** (5 files)
    * Location: ``src/AnalysisG/modules/typecasting``
    * Dependencies: structs, tools
    * Purpose: Type conversion utilities
    * Defines: ``variable_t``, ``write_t``, merge/sum/contract operations

**notification** (2 files)
    * Location: ``src/AnalysisG/modules/notification``
    * Dependencies: (none - foundation module)
    * Purpose: Logging and progress tracking
    * Note: Foundation module for messaging system

Visualization (3 modules)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Modules for plotting and result visualization.

**plotting** (2 files)
    * Location: ``src/AnalysisG/modules/plotting``
    * Dependencies: notification, structs, tools
    * Purpose: General plotting utilities

**roc** (2 files)
    * Location: ``src/AnalysisG/modules/roc``
    * Dependencies: plotting
    * Purpose: ROC curve generation

**metrics** (4 files)
    * Location: ``src/AnalysisG/modules/metrics``
    * Dependencies: notification, structs, templates
    * Purpose: Metric computation utilities

Physics (1 module)
~~~~~~~~~~~~~~~~~~

Specialized physics algorithms.

**nusol** (44 files)
    * Location: ``src/AnalysisG/modules/nusol``
    * Dependencies: conuix, ellipse, notification, reconstruction, structs, templates, tools
    * Purpose: Neutrino reconstruction algorithms
    * Submodules:
        - conuix: Constraint-based solver
        - ellipse: Ellipse parameterization
        - nusol: Main reconstruction algorithms
        - multisol: Multiple solution handling
    * Algorithms:
        - Single neutrino W→lν (analytical)
        - Double neutrino ttbar→lνblνb (numerical)
        - Matrix-based constraint solving

PyC - C++/CUDA/Python Interface (7 modules)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-performance CUDA kernels and C++/Python bindings.

**pyc/cutils** (6 files)
    * Location: ``src/AnalysisG/pyc/cutils``
    * Dependencies: (none - standalone)
    * Purpose: C++ utilities and conversions

**pyc/physics** (5 files)
    * Location: ``src/AnalysisG/pyc/physics``
    * Dependencies: (none - standalone)
    * Purpose: Physics calculations on GPU
    * Functions: P(), P2(), Beta(), M(), M2(), Mt(), Mt2(), Theta(), DeltaR()

**pyc/operators** (5 files)
    * Location: ``src/AnalysisG/pyc/operators``
    * Dependencies: (none - standalone)
    * Purpose: Vector operations on GPU
    * Functions: Dot(), Cross(), CosTheta(), SinTheta(), Rx(), Ry(), Rz()

**pyc/transform** (5 files)
    * Location: ``src/AnalysisG/pyc/transform``
    * Dependencies: (none - standalone)
    * Purpose: Coordinate transformations
    * Functions: PxPyPzE2PtEtaPhiE(), PtEtaPhiE2PxPyPzE()

**pyc/graph** (8 files)
    * Location: ``src/AnalysisG/pyc/graph``
    * Dependencies: (none - standalone)
    * Purpose: Graph algorithms on GPU
    * Functions: page_rank(), page_rank_reconstruction()

**pyc/nusol** (12 files)
    * Location: ``src/AnalysisG/pyc/nusol``
    * Dependencies: (none - standalone)
    * Purpose: GPU-accelerated neutrino reconstruction
    * Implements: CUDA kernels for analytical and numerical solvers

**pyc/interface** (7 files)
    * Location: ``src/AnalysisG/pyc/interface``
    * Dependencies: templates, tools
    * Purpose: Python/C++ interface layer

Events (7 variants)
~~~~~~~~~~~~~~~~~~~

Concrete event implementations for different analyses.

**events/bsm_4tops** (6 files)
    * Location: ``src/AnalysisG/events/bsm_4tops``
    * Dependencies: bsm_4tops, templates
    * Purpose: BSM 4-top search event class

**events/exp_mc20** (6 files)
    * Location: ``src/AnalysisG/events/exp_mc20``
    * Dependencies: exp_mc20, templates
    * Purpose: Experimental MC20 campaign events

**events/ssml_mc20** (8 files)
    * Location: ``src/AnalysisG/events/ssml_mc20``
    * Dependencies: ssml_mc20, templates
    * Purpose: Semi-supervised ML MC20 events

**events/gnn** (6 files)
    * Location: ``src/AnalysisG/events/gnn``
    * Dependencies: inference, templates
    * Purpose: GNN inference events

Graphs (3 variants)
~~~~~~~~~~~~~~~~~~~

Concrete graph implementations for GNN inputs.

**graphs/bsm_4tops** (9 files)
    * Location: ``src/AnalysisG/graphs/bsm_4tops``
    * Dependencies: bsm_4tops, templates
    * Purpose: Graph representation for 4-top analysis

**graphs/exp_mc20** (6 files)
    * Location: ``src/AnalysisG/graphs/exp_mc20``
    * Dependencies: exp_mc20, templates
    * Purpose: Graph representation for MC20 experimental data

**graphs/ssml_mc20** (6 files)
    * Location: ``src/AnalysisG/graphs/ssml_mc20``
    * Dependencies: ssml_mc20, templates
    * Purpose: Graph representation for SSML analysis

Metrics (2 implementations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Concrete metric implementations.

**metrics/accuracy** (4 files)
    * Location: ``src/AnalysisG/metrics/accuracy``
    * Dependencies: templates
    * Purpose: Accuracy metric for classification

**metrics/pagerank** (4 files)
    * Location: ``src/AnalysisG/metrics/pagerank``
    * Dependencies: templates
    * Purpose: PageRank metric for graph analysis

Models (2 implementations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Concrete model implementations.

**models/grift** (4 files)
    * Location: ``src/AnalysisG/models/grift``
    * Dependencies: templates
    * Purpose: GRaph Interaction FineTuning model

**models/RecursiveGraphNeuralNetwork** (3 files)
    * Location: ``src/AnalysisG/models/RecursiveGraphNeuralNetwork``
    * Dependencies: templates
    * Purpose: Recursive GNN architecture

Core (15 files)
~~~~~~~~~~~~~~~

**core** (15 files)
    * Location: ``src/AnalysisG/core``
    * Dependencies: (none - standalone)
    * Purpose: Core Python interface layer
    * Contains: All template base classes in Cython

Complete Dependency Graph
--------------------------

Foundation Modules (No Dependencies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These modules form the foundation and have no dependencies on other modules:

* **tools** - Base utilities
* **notification** - Logging system
* **core** - Python interface
* **pyc/cutils** - C++ utilities
* **pyc/physics** - GPU physics
* **pyc/operators** - GPU operators
* **pyc/transform** - GPU transforms
* **pyc/graph** - GPU graph algorithms
* **pyc/nusol** - GPU neutrino solver

Level 1 Dependencies (Depend only on Foundation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These modules depend only on foundation modules:

* **structs** → tools
* **plotting** → notification, structs, tools
* **meta** → notification, structs, tools

Level 2 Dependencies
~~~~~~~~~~~~~~~~~~~~~

* **typecasting** → structs, tools
* **io** → meta, notification, structs, tools
* **roc** → plotting

Level 3+ Dependencies (Complex)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **particle** → structs, templates, tools
* **graph** → structs, templates, tools
* **selection** → meta, structs, templates, tools
* **event** → meta, reconstruction, structs, templates, tools
* **model** → notification, structs, templates
* **metrics** → notification, structs, templates
* **lossfx** → notification, structs, templates, tools
* **metric** → meta, notification, plotting, structs, templates, tools
* **container** → generators, meta, templates, tools
* **optimizer** → generators, metrics, structs, templates
* **sampletracer** → container, generators, notification
* **dataloader** → generators, io, notification, structs, templates, tools
* **nusol** → conuix, ellipse, notification, reconstruction, structs, templates, tools
* **pyc/interface** → templates, tools
* **analysis** → AnalysisG, generators, io, structs, templates, tools

Circular Dependencies
---------------------

**None detected**. The module structure forms a proper directed acyclic graph (DAG).

Dependency Chain Analysis
--------------------------

Longest Dependency Chain
~~~~~~~~~~~~~~~~~~~~~~~~~

The longest dependency chain is through the **analysis** module::

    User Code
      ↓
    analysis
      ↓
    io, dataloader, container
      ↓
    meta, structs, templates
      ↓
    tools, notification

Maximum depth: 4 levels

Critical Path Modules
~~~~~~~~~~~~~~~~~~~~~~

These modules are dependencies for many other modules:

1. **tools** - 15 modules depend on it
2. **structs** - 14 modules depend on it
3. **templates** - 13 modules depend on it
4. **notification** - 10 modules depend on it
5. **meta** - 6 modules depend on it

Build Order Recommendation
---------------------------

Based on the dependency analysis, modules should be built in this order:

**Phase 1: Foundation** (parallel build possible)
    * tools
    * notification
    * core
    * pyc/cutils, pyc/physics, pyc/operators, pyc/transform, pyc/graph, pyc/nusol

**Phase 2: Core Structures** (after Phase 1)
    * structs

**Phase 3: Utilities** (after Phase 2)
    * typecasting
    * plotting
    * meta

**Phase 4: I/O and Templates** (after Phase 3)
    * io
    * roc

**Phase 5: Template Classes** (after Phase 4)
    * particle
    * graph (template)
    * selection
    * event
    * model
    * metrics (module)
    * lossfx
    * metric

**Phase 6: Infrastructure** (after Phase 5)
    * container
    * optimizer
    * pyc/interface

**Phase 7: Advanced** (after Phase 6)
    * sampletracer
    * dataloader
    * nusol

**Phase 8: Orchestration** (after Phase 7)
    * analysis

**Phase 9: Concrete Implementations** (after Phase 8, parallel build possible)
    * events/bsm_4tops, events/exp_mc20, events/ssml_mc20, events/gnn
    * graphs/bsm_4tops, graphs/exp_mc20, graphs/ssml_mc20
    * metrics/accuracy, metrics/pagerank
    * models/grift, models/RecursiveGraphNeuralNetwork

Module File Statistics
-----------------------

.. list-table:: Module File Counts
   :header-rows: 1
   :widths: 40 20 40

   * - Module
     - File Count
     - Category
   * - nusol
     - 44
     - Physics (largest module)
   * - structs
     - 18
     - Utilities
   * - core
     - 15
     - Core Python Interface
   * - pyc/nusol
     - 12
     - PyC CUDA/Python
   * - graphs/bsm_4tops
     - 9
     - Graphs
   * - analysis
     - 9
     - Analysis Infrastructure
   * - events/ssml_mc20
     - 8
     - Events
   * - pyc/graph
     - 8
     - PyC CUDA/Python
   * - pyc/interface
     - 7
     - PyC CUDA/Python
   * - events/bsm_4tops
     - 6
     - Events
   * - events/exp_mc20
     - 6
     - Events
   * - events/gnn
     - 6
     - Events
   * - graphs/exp_mc20
     - 6
     - Graphs
   * - graphs/ssml_mc20
     - 6
     - Graphs
   * - metric
     - 6
     - Core Templates
   * - particle
     - 6
     - Core Templates
   * - pyc/cutils
     - 6
     - PyC CUDA/Python
   * - io
     - 5
     - Data Management
   * - lossfx
     - 5
     - Analysis Infrastructure
   * - model
     - 5
     - Core Templates
   * - pyc/operators
     - 5
     - PyC CUDA/Python
   * - pyc/physics
     - 5
     - PyC CUDA/Python
   * - pyc/transform
     - 5
     - PyC CUDA/Python
   * - typecasting
     - 5
     - Utilities

Usage in Analysis
-----------------

Typical User Workflow
~~~~~~~~~~~~~~~~~~~~~

1. **Define Templates** (user extends):
   
   * EventTemplate → Custom event class
   * ParticleTemplate → Custom particle types
   * SelectionTemplate → Physics object selections
   * GraphTemplate → Graph structure for GNN
   * ModelTemplate → PyTorch model
   * MetricTemplate → Evaluation metrics

2. **Configure Analysis**:
   
   .. code-block:: python
   
       from AnalysisG import Analysis
       
       analysis = Analysis()
       analysis.add_event_template(MyEvent)
       analysis.add_selection_template(MySelection)
       analysis.add_graph_template(MyGraph)
       analysis.add_model(MyModel)
       analysis.add_metric_template(MyMetric)

3. **Execute Pipeline**:
   
   .. code-block:: python
   
       analysis.start()  # Orchestrates all modules

Internal Dependency Flow
~~~~~~~~~~~~~~~~~~~~~~~~

When ``analysis.start()`` is called, it coordinates these modules in order:

1. **io** - Read ROOT/HDF5 files
2. **event** - Build EventTemplate instances
3. **selection** - Apply SelectionTemplate
4. **graph** - Build GraphTemplate for GNN
5. **dataloader** - Create training batches
6. **model** - Execute PyTorch training
7. **optimizer** - Update model weights
8. **lossfx** - Compute loss and backprop
9. **metric** - Evaluate performance
10. **plotting**, **roc** - Generate visualizations
11. **io** (write) - Save results to ROOT/HDF5

Summary
-------

The AnalysisG framework consists of:

* **52 distinct modules**
* **339 source files** (124 headers, 215 sources)
* **11 logical categories**
* **4-level maximum dependency depth**
* **No circular dependencies**
* **9-phase recommended build order**

All modules are organized in a clean dependency hierarchy with foundation modules
at the bottom and high-level orchestration at the top. The modular design allows
for flexible extension while maintaining clear separation of concerns.

See Also
--------

* :doc:`/modules/core_templates/overview` - Core template classes
* :doc:`/technical/build/cmake` - CMake build system
* :doc:`/technical/analysis_workflow` - Analysis pipeline workflow
