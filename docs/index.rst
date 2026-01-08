.. AnalysisG documentation master file

==========================================
Welcome to AnalysisG's Documentation
==========================================

AnalysisG is a comprehensive framework for graph-based analysis in high-energy physics (HEP).
It provides a flexible and extensible architecture for processing physics event data, constructing
graph representations, applying machine learning models, and evaluating analysis results.

.. image:: https://readthedocs.org/projects/analysisg/badge/?version=latest
   :target: https://analysisg.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Getting Started
===============

To get started with AnalysisG, check out:

1. :doc:`source/installation` - Installation instructions
2. :doc:`source/quickstart` - Quick start guide
3. :doc:`source/api/library_root` - Full API reference

Key Features
============

- **Graph-based Analysis**: Transform physics events into graph structures for ML
- **PyTorch Integration**: Seamless integration with PyTorch for neural networks
- **Modular Design**: Extensible template-based architecture
- **ROOT/HDF5 Support**: Read from ROOT files, store in HDF5
- **Multi-threading**: Efficient parallel processing

Architecture Overview
=====================

The framework is organized into two main components:

**Core Module**
   Contains fundamental C++ implementations for graph structures, nodes, and edges.

**Modules**
   Specialized components extending framework functionality:
   
   - **Analysis**: Central workflow orchestration
   - **Event**: Event data structures
   - **Graph**: Graph construction
   - **Particle**: Particle representations
   - **Model**: Machine learning models
   - **Selection**: Event selection criteria
   - **Metric**: Performance evaluation

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   source/installation
   source/quickstart
   source/tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   source/api/library_root

.. toctree::
   :maxdepth: 1
   :caption: Development

   source/contributing
   source/changelog

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
