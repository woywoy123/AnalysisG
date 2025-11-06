AnalysisG Documentation
=======================

Welcome to the comprehensive documentation for AnalysisG, a high-performance analysis framework for High Energy Physics (HEP) featuring C++, CUDA, and Cython implementations.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   introduction
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Simple Interfaces (User-Overridable):

   interfaces/overview
   interfaces/core_templates
   interfaces/event_interface
   interfaces/particle_interface
   interfaces/graph_interface
   interfaces/metric_interface
   interfaces/model_interface
   interfaces/selection_interface

.. toctree::
   :maxdepth: 2
   :caption: Complex Technical Components:

   technical/overview
   technical/cpp_modules_api
   technical/cuda_kernels_api
   technical/cpp_api
   technical/cuda_api

.. toctree::
   :maxdepth: 2
   :caption: Core Package:

   core/analysis
   core/event_template
   core/particle_template
   core/graph_template
   core/metric_template
   core/model_template
   core/selection_template
   core/io
   core/lossfx
   core/meta
   core/notification
   core/plotting
   core/roc
   core/structs
   core/tools

.. toctree::
   :maxdepth: 2
   :caption: Events Package:

   events/overview
   events/bsm_4tops
   events/exp_mc20
   events/gnn
   events/ssml_mc20

.. toctree::
   :maxdepth: 2
   :caption: Graphs Package:

   graphs/overview
   graphs/bsm_4tops
   graphs/exp_mc20
   graphs/ssml_mc20

.. toctree::
   :maxdepth: 2
   :caption: Metrics Package:

   metrics/overview
   metrics/accuracy
   metrics/pagerank

.. toctree::
   :maxdepth: 2
   :caption: Models Package:

   models/overview
   models/grift
   models/rgnn

.. toctree::
   :maxdepth: 2
   :caption: Modules Package (C++ Implementation):

   modules/overview
   modules/analysis
   modules/container
   modules/dataloader
   modules/event
   modules/graph
   modules/io
   modules/lossfx
   modules/meta
   modules/metric
   modules/metrics
   modules/model
   modules/notification
   modules/nusol
   modules/optimizer
   modules/particle
   modules/plotting
   modules/roc
   modules/sampletracer
   modules/selection
   modules/structs
   modules/tools
   modules/typecasting

.. toctree::
   :maxdepth: 2
   :caption: PyC Package (C++/CUDA/Cython):

   pyc/overview
   pyc/cutils
   pyc/graph
   pyc/interface
   pyc/nusol
   pyc/operators
   pyc/physics
   pyc/transform

.. toctree::
   :maxdepth: 2
   :caption: Templates:

   templates/overview
   templates/events
   templates/graphs
   templates/metrics
   templates/models
   templates/particles

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
