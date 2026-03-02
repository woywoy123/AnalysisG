AnalysisG Documentation
=======================

**AnalysisG** is a Graph Neural Network Analysis Framework for High Energy Physics.
It provides a complete pipeline for translating ROOT n-tuples into graph-structured
data, training and evaluating Graph Neural Networks, and running cut-based
selections — all from a Python interface backed by high-performance C++ and CUDA.

.. toctree::
   :maxdepth: 1
   :caption: Overview

   introduction
   installation

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   quick_start

.. toctree::
   :maxdepth: 1
   :caption: Python Interface

   api/core/analysis
   api/core/particle_template
   api/core/event_template
   api/core/graph_template
   api/core/selection_template
   api/core/metric_template
   api/core/model_template
   api/core/io
   api/core/meta
   api/core/tools
   api/core/notification
   api/core/plotting
   api/core/roc
   api/core/lossfx
   api/core/structs

.. toctree::
   :maxdepth: 1
   :caption: C++ User Templates

   api/modules/particle
   api/modules/event
   api/modules/graph
   api/modules/selection
   api/modules/metric
   api/modules/model

.. toctree::
   :maxdepth: 1
   :caption: C++ Framework

   api/modules/analysis
   api/modules/container
   api/modules/sampletracer
   api/modules/dataloader
   api/modules/optimizer
   api/modules/io
   api/modules/meta
   api/modules/tools
   api/modules/notification
   api/modules/plotting
   api/modules/roc
   api/modules/lossfx
   api/modules/metrics
   api/modules/structs
   api/modules/typecasting

.. toctree::
   :maxdepth: 1
   :caption: Neutrino Solutions (NuSol)

   api/modules/nusol

.. toctree::
   :maxdepth: 1
   :caption: pyc — CUDA Extensions

   api/pyc/interface
   api/pyc/physics
   api/pyc/transform
   api/pyc/graph
   api/pyc/operators
   api/pyc/nusol
   api/pyc/cutils

.. toctree::
   :maxdepth: 1
   :caption: About

   changelog

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
