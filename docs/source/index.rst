AnalysisG Documentation
=======================

**AnalysisG** is a Graph Neural Network Analysis Framework for High Energy Physics.
It provides a complete pipeline for translating ROOT n-tuples into graph-structured
data, training and evaluating Graph Neural Networks, and running cut-based
selections — all from a Python interface backed by high-performance C++ and CUDA.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   introduction
   installation
   quick_start

.. toctree::
   :maxdepth: 1
   :caption: Python User Guide

   api/core/analysis
   api/core/particle_template
   api/core/event_template
   api/core/graph_template
   api/core/selection_template
   api/core/model_template
   api/core/metric_template
   api/core/advanced_api

.. toctree::
   :maxdepth: 1
   :caption: Standard Library

   api/events/index
   api/graphs/index
   api/models/index
   api/metrics/index
   api/selections/index

.. toctree::
   :maxdepth: 1
   :caption: Advanced / C++ Backend

   api/modules/index_templates
   api/modules/index_framework
   api/modules/nusol
   api/pyc/index

.. toctree::
   :maxdepth: 1
   :caption: About

   changelog

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
