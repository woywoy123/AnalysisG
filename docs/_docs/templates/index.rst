.. _core_concepts:

=============
Core Concepts
=============

AnalysisG is built upon several core concepts that work together to enable HEP data analysis using Graph Neural Networks.

Architecture Overview
---------------------

.. code-block:: text

    ROOT Data → Events → Graphs → Trained Models
         │         │        │          │
         ▼         ▼        ▼          ▼
    IO System → Event Templates → Graph Templates → Model Templates

The main components are:

1.  **Event Templates**: Read and organize data from ROOT files
2.  **Particle Templates**: Define the properties of physical objects
3.  **Graph Templates**: Transform event data into graph structures
4.  **Model Templates**: Define and train neural network models
5.  **Dataloader**: Manages data for training and evaluation





From Physics Event to Graph
---------------------------

A key concept in AnalysisG is the transformation of physics events into graph structures:

*   **Event → Graph Nodes**: Particles (jets, electrons, etc.) become nodes in the graph
*   **Particle Interactions → Graph Edges**: Relationships between particles become edges
*   **Particle Properties → Node Features**: PT, Eta, Phi, etc. become features
*   **Interaction Properties → Edge Features**: ΔR, ΔΦ, etc. become features
*   **Event Properties → Graph Features**: MET, HT, etc. become global features

This transformation is defined by the ``graph_template`` class.

Data Chain
----------

Data processing in AnalysisG follows a clear chain:

1.  **Data Input**: ROOT files are read by the IO system
2.  **Event Building**: Event templates extract and organize the data
3.  **Graph Creation**: Graph templates convert events into graphs
4.  **Caching**: The created graphs are stored for later use
5.  **Training**: The graphs are used to train GNN models
6.  **Inference**: The trained models are applied to new data

Feature Engineering
-------------------

AnalysisG offers various ways to define features:

*   **Direct Features**: Directly from ROOT file data
*   **Calculated Features**: Based on physics formulas
*   **Relational Features**: Calculated from relationships between particles
*   **Truth Features**: From MC truth information
*   **Data Features**: From reconstructed measurements

Each feature type can be defined at different levels:

*   **Node Features**: For individual particles
*   **Edge Features**: For relationships between particles
*   **Graph Features**: For the entire event

This flexible feature engineering structure allows complex physics concepts to be transformed into trainable features.

Template System
---------------

A central design principle of AnalysisG is the template system, which provides flexibility and extensibility:

*   **Templates Enable Abstraction**: Physics concepts are abstracted into reusable components
*   **Hierarchical Organization**: Templates can be inherited and specialized
*   **Consistent Interfaces**: Standardized methods for interaction between components
*   **Extensibility**: Users can create new templates without modifying the core code

The main advantages of this system are:

1.  **Modularity**: Components can be developed and tested independently
2.  **Reusability**: Templates can be reused across different analyses
3.  **Customizability**: Users can adapt the framework to their specific requirements

High-Performance Architecture
-----------------------------

AnalysisG utilizes a hybrid architecture for maximum performance:

*   **C++ Backend**: Computationally intensive operations are implemented in C++
*   **Python Frontend**: User-friendly interface for analysis definition
*   **Cython Bindings**: Seamless integration between Python and C++
*   **PyTorch Integration**: Powerful tensor operations and GPU acceleration

This architecture enables:

*   **Efficient Processing of Large Datasets**: Optimized for millions of events
*   **GPU Acceleration**: For both data processing and training
*   **Interactive Development**: Python interface for rapid iterations
*   **Production-Ready Performance**: C++ backend for stable deployments

Summary
-------

The AnalysisG framework was designed from the ground up for modern HEP data analysis. It combines:

*   **Domain-Specific Design**: Optimized for HEP use cases
*   **Graph-Based Modeling**: Ideal for capturing complex particle relationships
*   **High-Performance Computing**: For processing large experiments
*   **User-Friendly Interfaces**: For maximum productivity

These core concepts form the foundation for all components of the framework, which are described in detail in the following sections.



Core Concepts
=============

Understanding these core concepts is key to using AnalysisG effectively.

EventTemplate & ParticleTemplate
--------------------------------

*   **`ParticleTemplate`**: Defines the properties and methods associated with a specific type of particle in your analysis (e.g., Jet, Electron, Muon, TruthTop). You typically inherit from `particle_template` and add methods to access particle data from your input source (like ROOT TTree leaves).
*   **`EventTemplate`**: Represents a single physics event. You inherit from `event_template` to define which particles are present in the event, how to access event-level information (like MET, weights), and how to link particles together. It reads data entry by entry from the input files.

See also: :doc:`/src/AnalysisG/templates/events/eventtemplate` (Note: Link might need adjustment based on final doc structure)

GraphTemplate
-------------

*   **`GraphTemplate`**: Defines how to convert an `EventTemplate` instance into a graph structure suitable for GNNs. You inherit from `graph_template` and implement the `CompileEvent` method.
*   **Features**: Within `CompileEvent`, you specify functions (lambdas) to extract:
    *   **Node Features**: Properties of individual particles (nodes).
    *   **Edge Features**: Properties relating pairs of particles (edges).
    *   **Graph Features**: Global properties of the event (graph-level attributes).
    *   **Truth Features**: Target labels for training (node-level, edge-level, or graph-level).
*   **Topology**: You can define criteria for edge creation (e.g., connect only particles within a certain Delta R) or use a fully connected graph.

See also: :doc:`graphtemplate`

Analysis Orchestration
----------------------

*   The `analysis` class manages the overall workflow. You configure it with:
    *   Input data samples.
    *   Registered `EventTemplate` and `GraphTemplate` implementations.
    *   Settings like number of threads, output paths, caching options.
*   It handles reading data, processing events using your templates, building graphs, and potentially managing GNN training loops via the `optimizer` and `dataloader`.

Dataloader, Optimizer, Model
----------------------------

*   **`dataloader`**: Takes the generated graphs and prepares them in batches suitable for GNN training or inference, handling aspects like shuffling and k-folding.
*   **`optimizer`**: Manages the training loop for GNN models, including applying gradients, scheduling learning rates, and interacting with loss functions.
*   **`model_template`**: A base class for defining your GNN architecture within the C++ framework, although often models are defined directly in Python/PyTorch.

IO and Metadata
---------------

*   **`io`**: Handles reading ROOT files efficiently and provides an interface for saving/loading processed data (like graphs or analysis tags) often using HDF5.
*   **`meta`**: Extracts and stores metadata associated with datasets (e.g., cross-sections, Monte Carlo generator information, sum of weights) often by reading specific ROOT trees or using tools like PyAMI.


Introduction to AnalysisG
======================

AnalysisG is a physics analysis framework designed for high-energy particle physics, with a particular focus on optimizing workflows for analyzing data from the ATLAS experiment at CERN's Large Hadron Collider.

Framework Philosophy
-------------------

The framework is built on a few key principles:

1. **Graph-based approach**: Physics events are represented as graphs, where particles are nodes and their interactions are edges, enabling advanced machine learning techniques
2. **Modular design**: Components are loosely coupled, allowing you to use just what you need
3. **Performance-oriented**: C++ core with Python bindings for performance-critical operations

Key Features
-----------

* **Event handling**: Efficient reading and processing of event data from ROOT files
* **Physics object reconstruction**: Tools for particle identification and reconstruction
* **Graph-based analytics**: Specialized data structures for applying machine learning to physics data
* **Selection framework**: Define and apply physics region selections
* **Integrated machine learning**: Ready-to-use ML interfaces for physics analysis
* **Meta-data handling**: Tracking cross-sections, weights, and other experimental parameters

Framework Components
------------------

.. figure:: ../images/framework_structure.png
   :width: 600px
   :align: center
   :alt: AnalysisG Framework Components
   
   The key components and their relationships in the AnalysisG framework

The framework is organized into several key components:

* **Core**: The central analysis object and base functionality
* **Modules**: I/O, containers, meta-data handling, and other utilities
* **Events**: Event templates for different physics processes and experiments
* **Graphs**: Graph representations of physics events for machine learning
* **Selections**: Physics selection regions and criteria

Use Cases
--------

AnalysisG is particularly well-suited for:

* Top quark physics analyses
* BSM (Beyond Standard Model) searches
* Machine learning applications in particle physics
* Neutrino reconstruction
* Performance benchmarking of physics algorithms

























====================
Examples
====================

This page contains a collection of examples demonstrating various aspects of AnalysisG. The examples range from basic concepts to advanced applications.

Framework Examples
================

This section contains examples from the ``studies`` directory of the AnalysisG framework, organized by topic. These examples demonstrate how to use the framework for various physics analyses.

.. toctree::
   :maxdepth: 1
   :caption: Example Categories
   
   analysis/index
   benchmarks/index
   mc16_matching/index
   mc20_experimental/index
   neutrino/index
   performance/index
   topreconstruction/index

Example Overview
--------------

Each example includes:

* Source file information
* Import statements
* Class definitions
* Key functions
* Entry points or main execution blocks

Basic Examples
-------------

Below are some fundamental examples to help you get started with AnalysisG:

Event Creation and Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import analysisg as ag
    
    # Create analysis instance
    analysis = ag.Analysis()
    
    # Add input files
    analysis.add_samples("data/mc16/*.root", "ttbar")
    
    # Register event template
    analysis.add_event_template(ag.ssml_mc20(), "event")
    
    # Start processing
    analysis.start()

Graph Construction
~~~~~~~~~~~~~~~~

.. code-block:: python

    import analysisg as ag
    
    # Create analysis instance
    analysis = ag.Analysis()
    
    # Add input files
    analysis.add_samples("data/mc16/*.root", "ttbar")
    
    # Register event and graph templates
    analysis.add_event_template(ag.ssml_mc20(), "event")
    analysis.add_graph_template(ag.graph_jets(), "jets")
    
    # Start processing
    analysis.start()

Selection Application
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import analysisg as ag
    
    # Create analysis instance
    analysis = ag.Analysis()
    
    # Add input files
    analysis.add_samples("data/mc16/*.root", "ttbar")
    
    # Register event template and selection
    analysis.add_event_template(ag.ssml_mc20(), "event")
    analysis.add_selection_template(ag.regions())
    
    # Start processing
    analysis.start()

Example Code Structure
--------------------

Examples are organized by physics topic and follow a consistent structure:

.. code-block:: none

    examples/
    ├── analysis/
    │   ├── basic_analysis_example.py
    │   ├── ttbar_analysis_example.py
    │   └── ...
    ├── benchmarks/
    │   ├── performance_benchmark_example.py
    │   └── ...
    ├── mc16_matching/
    │   └── ...
    └── ...

Running Examples
--------------

You can run these examples with:

.. code-block:: bash

    python /path/to/example.py

For more details on a specific category of examples, click on the corresponding section in the navigation.

Further Resources
-----------------

* The GitHub repository contains more [examples](https://github.com/woywoy123/AnalysisG/tree/master/examples).
* The :ref:`Tutorials <tutorials/index>` provide more detailed step-by-step guides.
* The :doc:`API Reference </api_reference/index>` contains technical details on all classes and methods.



Introduction to EventTemplates and ParticleTemplates
----------------------------------------------------

Given that C++ is a compiled language, the compiler needs to know about the source files which hold the definitions of the objects.
To achieve this, the package uses `CMake` to link everything together and make the package available for the compiler.

The Source Files 
^^^^^^^^^^^^^^^^

But first there are some files that need to be created for the event and particles to be defined.

- Create a new folder within `src/AnalysisG/events/`.
- Within the folder create the following files:

  - `CMakeLists.txt` (copy the one from `bsm_4tops` for example)
  - `event.cxx`
  - `event.h`
  - `particles.cxx`
  - `particles.h`
- Outside of the event folder, create the following files:

  - `event_<some name>.pxd`
  - `event_<some name>.pyx`
- Modify the `CMakeLists.txt` within the events folder and add the event to the list.

TLDR (Too Long Do Read)
-----------------------

For a quick introduction by example, checkout the template code under `src/AnalysisG/templates`.
A brief explanation:

- **events**: Templates used to define an event implementation.
- **graphs**: Templates used to define graphs for GNN training and inference.
- **selections**: Templates used for indepth studies using the associated event implementation.
- **model**: Templates for implementing a custom GNN model.
