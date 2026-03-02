Introduction
============

Abstract
--------

As the field of High Energy Particle Physics (HEPP) has begun exploring more
exotic machine learning algorithms, such as Graph Neural Networks (GNNs),
analyses commonly rely on pre-existing data science frameworks — including
PyTorch, TensorFlow and Keras — to recast ROOT samples into an appropriate
data structure.  This often results in tedious and computationally expensive
co-routines.

**AnalysisG** addresses these issues by following a similar philosophy to
*AnalysisTop*: events and particles are treated as polymorphic objects.  The
framework translates ROOT n-tuples into user-defined particle and event
objects, matches particles within complex decay chains, and constructs graph
structures with edge, node and graph-level feature tensors ready for GNN
training or inference.

For cut-based analyses the framework provides selection templates that accept
event objects, perform detailed studies, and export results to ROOT n-tuples or
serialised plot objects.

To facilitate fast machine learning in HEP, a self-contained sub-package
called **pyc** (*Python CUDA*) implements high-performance C++ and CUDA
kernels via the LibTorch API.  These include :math:`\Delta R`, polar/Cartesian
transforms, invariant-mass computation, edge/node aggregation, and analytical
single/double neutrino reconstruction.

Core Modules
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Description
   * - **EventTemplate**
     - Template class for user-defined event and particle definitions.
   * - **ParticleTemplate**
     - Template class used with *EventTemplate* to define particle types.
   * - **GraphTemplate**
     - Defines inclusive graph features (edge, node, global attributes).
   * - **SelectionTemplate**
     - Template for custom event-selection algorithms.
   * - **MetricTemplate**
     - Template for ML evaluation metrics.
   * - **ModelTemplate**
     - Template for GNN model definitions.
   * - **Plotting**
     - Wrapper around boost_histograms and mpl-hep for plot definitions.
   * - **io**
     - Cython interface to CERN ROOT for reading n-tuples in 3 lines.
   * - **MetaData**
     - DSID search and data-scraping via a modified PyAMI interface.
   * - **Analysis**
     - Main analysis compiler that chains user-defined template actions.
   * - **Tools**
     - Utility class (file system, string, math helpers) used throughout.
   * - **pyc**
     - Self-contained C++/CUDA sub-package for HEP-specific kernels.

See :doc:`quick_start` for a step-by-step walkthrough with code examples.

Languages and Technologies
--------------------------

* **C++20** — core engine, modules, CUDA wrappers.
* **Cython** — Python/C++ bridge with minimal overhead.
* **CUDA** — GPU kernels for physics computations.
* **LibTorch** — tensor operations inside CUDA kernels.
* **Doxygen + Breathe + Sphinx** — documentation pipeline.
