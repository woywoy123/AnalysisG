AnalysisG Documentation
========================

.. meta::
   :description: AnalysisG - Graph Neural Network Framework for High Energy Physics Analysis
   :keywords: HEP, particle physics, graph neural networks, GNN, machine learning, PyTorch, ROOT, C++

Welcome to AnalysisG's documentation!

AnalysisG is a comprehensive C++ framework designed for High Energy Physics (HEP) analysis,
specifically focusing on Graph Neural Network (GNN) applications for particle physics data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   quickstart
   user_guide
   api_reference
   examples
   contributing

Introduction
============

AnalysisG provides a complete workflow for:

* Event processing and particle management
* Graph representation for GNN applications
* Data loading with k-fold cross-validation
* Machine learning model training with PyTorch
* Performance metrics and evaluation
* Visualization and plotting

Key Features
------------

* **Flexible Event System**: Customizable event templates for different analysis types
* **Particle Physics**: Full 4-vector support with automatic coordinate transformations
* **Graph Neural Networks**: Native support for GNN with PyTorch integration
* **Data Management**: Efficient data loader with batching and k-fold CV
* **Model Training**: Comprehensive loss functions and optimizers
* **I/O Support**: ROOT and HDF5 file formats
* **Selection Framework**: Powerful event and object selection tools

Quick Start
===========

Basic usage example:

.. code-block:: cpp

   #include <AnalysisG/analysis.h>
   
   // Create analysis instance
   analysis my_analysis;
   
   // Add data samples
   my_analysis.add_samples("/path/to/data", "signal");
   
   // Add event template
   my_event* evt = new my_event();
   my_analysis.add_event_template(evt, "default");
   
   // Add selection
   my_selection* sel = new my_selection();
   my_analysis.add_selection_template(sel);
   
   // Run analysis
   my_analysis.start();

Modules Overview
================

Analysis Module
---------------

.. doxygengroup:: analysis_module
   :content-only:

Particle Module
---------------

.. doxygengroup:: particle_module
   :content-only:

Event Module
------------

.. doxygengroup:: event_module
   :content-only:

Graph Module
------------

.. doxygengroup:: graph_module
   :content-only:

DataLoader Module
-----------------

.. doxygengroup:: dataloader_module
   :content-only:

Model Module
------------

.. doxygengroup:: model_module
   :content-only:

Selection Module
----------------

.. doxygengroup:: selection_module
   :content-only:

I/O Module
----------

.. doxygengroup:: io_module
   :content-only:

Loss Functions Module
---------------------

.. doxygengroup:: lossfx_module
   :content-only:

Metric Module
-------------

.. doxygengroup:: metric_module
   :content-only:

Container Module
----------------

.. doxygengroup:: container_module
   :content-only:

Meta Module
-----------

.. doxygengroup:: meta_module
   :content-only:

Notification Module
-------------------

.. doxygengroup:: notification_module
   :content-only:

Tools Module
------------

.. doxygengroup:: tools_module
   :content-only:

Structs Module
--------------

.. doxygengroup:: structs_module
   :content-only:

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
