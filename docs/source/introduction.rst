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
   * - **ParticleTemplate**
     - C++ base class for user-defined particles.  Provides kinematic
       properties (``pt``, ``eta``, ``phi``, ``e``, ``px``, ``py``, ``pz``,
       ``Mass``), classification flags (``is_b``, ``is_lep``, ``is_nu``,
       ``is_add``), decay-tree bookkeeping (``Children``, ``Parents``), and
       the ``add_leaf`` / ``apply_type_prefix`` / ``build`` interface for ROOT
       branch mapping.
   * - **EventTemplate**
     - C++ base class for user-defined physics events.  Declares the ROOT
       trees/branches to read (``trees``, ``add_leaf``), registers particle
       collections (``register_particle``), and provides ``build`` and
       ``CompileEvent`` hooks for constructing the event from raw branch data.
   * - **GraphTemplate**
     - C++ base class for graph construction.  Inside ``CompileEvent`` users
       call ``define_particle_nodes``, ``add_node_data_feature``,
       ``add_edge_truth_feature`` etc. to assemble the graph tensors that the
       ML pipeline consumes.
   * - **SelectionTemplate**
     - Template for custom cut-based event selections.  Provides
       ``dump``/``load`` serialisation, ``GetMetaData``, ``InterpretROOT``,
       and ``Postprocessing`` hooks.
   * - **MetricTemplate**
     - Template for ML evaluation metrics.  Exposes ``RunNames`` (dict),
       ``Variables`` (list), ``Postprocessing``, and ``InterpretROOT``.
   * - **ModelTemplate**
     - C++ base class for GNN model definitions.  Users override ``forward``
       to fetch tensors from a ``graph_t`` object and write predictions back.
   * - **OptimizerConfig**
     - Configuration struct for PyTorch optimizers (Adam, SGD, RMSprop,
       Adagrad, LBFGS) and learning-rate schedulers (StepLR, CyclicLR,
       ExponentialLR).  Passed to ``Analysis.AddModel``.
   * - **IO**
     - C++ class (inheriting ``tools`` + ``notification``) for reading CERN
       ROOT n-tuples.  Iterable in Python; each iteration yields a ``dict``
       whose keys are ``bytes`` in the format ``b'tree.leaf.leaf'``.
   * - **Analysis**
     - Top-level Python pipeline compiler.  Chains
       ``AddSamples`` / ``AddEvent`` / ``AddGraph`` / ``AddModel`` /
       ``AddSelection`` / ``AddMetric`` registrations and launches the full
       pipeline with ``Start()``.
   * - **Meta / MetaLookup**
     - ATLAS dataset metadata (AMI) cache and lookup helpers.  ``Meta``
       stores per-dataset fields (DSID, cross-section, generator, …);
       ``MetaLookup`` aggregates them and computes luminosity-weighted yields.
   * - **Plotting**
     - Python histogram and line-plot wrappers (``TH1F``, ``TH2F``, ``TLine``,
       ``ROC``) built on ``mplhep`` and boost-histogram.
   * - **Tools**
     - Utility class (file-system, string, hashing, math helpers) used
       throughout the framework.
   * - **pyc**
     - Self-contained C++/CUDA sub-package for HEP-specific PyTorch custom
       operators: :math:`\Delta R`, polar/Cartesian transforms, invariant-mass
       computation, edge/node aggregation, and neutrino reconstruction.

.. note::

   **Verified sample statistics** — as a sanity check the IO class was run
   against the dilepton test sample shipped with the repository:

   * File: ``test/samples/dilepton/DAOD_TOPQ1.21955717._000001.root``
   * Tree: ``nominal``
   * Events: **1,098**
   * Total jets (``b'nominal.jet_pt.jet_pt'``): **8,161**
   * Average jets per event: **7.43** (min 4, max 14)

   This result was obtained by iterating the ``IO`` class over the file and
   summing ``len(entry[b'nominal.jet_pt.jet_pt'])`` across all events.

See :doc:`quick_start` for a step-by-step walkthrough with code examples.

Languages and Technologies
--------------------------

* **C++20** — core engine, modules, CUDA wrappers.
* **Cython** — Python/C++ bridge with minimal overhead.
* **CUDA** — GPU kernels for physics computations.
* **LibTorch** — tensor operations inside CUDA kernels.
* **Doxygen + Breathe + Sphinx** — documentation pipeline.
