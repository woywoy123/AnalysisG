Modules
=======

AnalysisG is organized into several core modules, each with specific functionality.

.. toctree::
   :maxdepth: 1

   modules/analysis
   modules/graph
   modules/model
   modules/io
   modules/meta
   modules/metrics
   modules/plotting
   modules/particle
   modules/selection
   modules/structs
   modules/tools

Module Overview
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Description
   * - Analysis
     - Orchestrates the entire workflow, connecting all other components together. Provides high-level functionality for experiment execution.
   * - Graph
     - Handles graph representation, topology definition, and feature extraction from physical events.
   * - Model
     - Defines machine learning models, training routines, and inference infrastructure for neural networks.
   * - IO
     - Manages reading and writing data in various formats (ROOT, HDF5, etc.) common in high energy physics.
   * - Meta
     - Maintains metadata and configuration settings for analysis tasks, including experiment parameters.
   * - Metrics
     - Implements performance evaluation and metric tracking for model evaluation and physics results.
   * - Plotting
     - Provides visualization tools for analysis results and model performance metrics.
   * - Particle
     - Defines particle physics objects and their properties for use in event analysis.
   * - Selection
     - Implements event selection algorithms and criteria for filtering relevant physics events.
   * - Structs
     - Offers fundamental data structures used throughout the framework for data organization.
   * - Tools
     - Provides general utility functions for various tasks across the framework.

Architecture Diagram
------------------

.. image:: ../_static/architecture.png
   :width: 800
   :alt: AnalysisG Architecture

Module Dependencies
------------------

The diagram below shows the primary dependencies between modules:

.. code-block:: text

    Analysis
    ├── Graph
    │   ├── Particle
    │   └── Structs
    ├── Model
    │   ├── IO
    │   └── Metrics
    ├── Meta
    ├── Plotting
    │   └── Metrics
    ├── Selection
    │   └── Particle
    └── Tools