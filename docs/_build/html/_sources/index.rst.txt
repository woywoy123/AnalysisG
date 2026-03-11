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
   :caption: Built-In Events

   api/events/bsm_4tops
   api/events/exp_mc20
   api/events/gnn
   api/events/ssml_mc20

.. toctree::
   :maxdepth: 1
   :caption: Built-In Graphs

   api/graphs/bsm_4tops
   api/graphs/exp_mc20
   api/graphs/ssml_mc20

.. toctree::
   :maxdepth: 1
   :caption: Built-In Models

   api/models/grift
   api/models/rgnn

.. toctree::
   :maxdepth: 1
   :caption: Built-In Metrics

   api/metrics/accuracy
   api/metrics/pagerank

.. toctree::
   :maxdepth: 2
   :caption: Built-In Selections

   api/selections/example_met
   api/selections/analysis_regions
   api/selections/mc16_childrenkinematics
   api/selections/mc16_decaymodes
   api/selections/mc16_met
   api/selections/mc16_parton
   api/selections/mc16_topjets
   api/selections/mc16_topkinematics
   api/selections/mc16_topmatching
   api/selections/mc16_toptruthjets
   api/selections/mc16_zprime
   api/selections/mc20_matching
   api/selections/mc20_topkinematics
   api/selections/mc20_topmatching
   api/selections/mc20_zprime
   api/selections/neutrino_combinatorial
   api/selections/neutrino_validation
   api/selections/performance_topefficiency

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
