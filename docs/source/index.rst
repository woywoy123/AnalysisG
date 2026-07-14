AnalysisG Documentation
=======================

**AnalysisG** is a Graph Neural Network Analysis Framework for High Energy Physics.
It provides a complete pipeline for translating ROOT n-tuples into graph-structured
data, training and evaluating Graph Neural Networks, and running cut-based
selections — all from a Python interface backed by high-performance C++ and CUDA.

.. toctree::
   :maxdepth: 1
   :caption: Beginner

   introduction
   installation
   quick_start
   tutorial
   api/core/minimal_working_example

.. toctree::
   :maxdepth: 1
   :caption: Templates

   api/core/event_template
   api/core/particle_template
   api/core/graph_template
   api/core/selection_template
   api/core/model_template
   api/core/metric_template

.. toctree::
   :maxdepth: 1
   :caption: Core Class types

   api/modules/meta
   api/modules/plotting
   api/modules/graph

.. toctree::
   :maxdepth: 1
   :caption: Core Struct types

   api/modules/structs
   api/modules/event

.. toctree::
   :maxdepth: 1
   :caption: Backend Containers

   api/modules/container
   api/modules/sampletracer
   api/modules/dataloader

.. toctree::
   :maxdepth: 1
   :caption: Reconstruction Algorithms

   api/modules/nusol
   api/metrics/pagerank

.. toctree::
   :maxdepth: 1
   :caption: Configurations

   api/modules/lossfx
   api/modules/optimizer

.. toctree::
   :maxdepth: 1
   :caption: Translations

   api/modules/typecasting
   api/modules/mergecast

.. toctree::
   :maxdepth: 1
   :caption: Toolings

   api/modules/notification
   api/modules/tools
   api/modules/io
   api/core/plotting

.. toctree::
   :maxdepth: 1
   :caption: Interfaces

   api/interfaces_cython

.. toctree::
   :maxdepth: 1
   :caption: CMake Tooling

   api/cmake_cybuild

.. toctree::
   :maxdepth: 1
   :caption: Exports

   api/exports_hdf5
   api/exports_root

.. toctree::
   :maxdepth: 1
   :caption: Custom Maps

   api/custom_root_pcm
   api/custom_root_types
   api/custom_tensor_types
   api/custom_tensor_casting

.. toctree::
   :maxdepth: 1
   :caption: pyc: PyCUDA

   api/pyc/cutils
   api/pyc/physics
   api/pyc/graph
   api/pyc/missing_cuda

.. toctree::
   :maxdepth: 1
   :caption: Default Implementations

   api/events/index
   api/graphs/index
   api/selections/index
   api/models/index
   api/metrics/index

.. toctree::
   :maxdepth: 1
   :caption: About

   changelog

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
