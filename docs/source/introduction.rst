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

Usage Workflow
--------------

A typical AnalysisG analysis follows these steps:

1. **Define particles** — subclass :class:`particle_template`, set
   ``this->type`` to a unique string, call ``add_leaf("key", "root_branch")``
   for each ROOT leaf, call ``apply_type_prefix()``, and override
   ``build(std::map<std::string, particle_template*>* prt, element_t* el)``
   to populate the output map with heap-allocated instances.

2. **Define events** — subclass :class:`event_template`, set ``this->name``,
   call ``register_particle<MyParticle>(&m_jets)`` for each particle
   collection, set ``this->trees = {"nominal"}``, and override ``build`` /
   ``CompileEvent`` to populate public particle vectors from the private maps.

3. **Define graphs** — subclass :class:`graph_template`, set ``this->name``,
   and override ``CompileEvent``.  Inside ``CompileEvent`` call
   ``get_event<MyEvent>()`` to retrieve the populated event object, pass a
   particle collection to ``define_particle_nodes``, then register feature
   functions::

     // standalone feature functions follow the signature:
     //   void fn_name(OutputType* out, ParticleOrEventType* in)
     void pt(double* o, particle_template* p){*o = p->pt;}
     void signal(bool* o, MyEvent* ev){*o = ev->truth_signal;}
     void same_top(int* o, std::tuple<particle_template*, particle_template*>* e_ij){
         *o = std::get<0>(*e_ij)->index == std::get<1>(*e_ij)->index;
     }

     void MyGraph::CompileEvent(){
         MyEvent* ev = this->get_event<MyEvent>();
         this->define_particle_nodes(&ev->Jets);

         this->add_graph_truth_feature<bool, MyEvent>(ev, signal, "signal");
         this->add_node_data_feature<double, particle_template>(pt, "pt");
         this->add_edge_truth_feature<int, particle_template>(same_top, "top_edge");
     }

4. **Define selections** *(optional)* — subclass :class:`selection_template`
   and implement ``selection`` / ``strategy``.

5. **Wire up the pipeline** (Python interface):

   .. code-block:: python

      from AnalysisG import Analysis
      from AnalysisG.core.lossfx import OptimizerConfig

      op = OptimizerConfig()
      op.Optimizer = "adam"
      op.lr = 1e-3

      ana = Analysis()
      ana.OutputPath = "./output"
      ana.Epochs     = 20
      ana.AddSamples("./data/sample.root", "ttbar")
      ana.AddEvent(MyEvent(), "ttbar")
      ana.AddGraph(MyGraph(), "ttbar")
      ana.AddModel(MyModel(), op, "run1")
      ana.Start()

``pyc`` kernels are exposed as PyTorch custom operators and can be called
directly after loading the shared library:

.. code-block:: python

   import torch
   torch.ops.load_library("libtpyc.so")   # CPU build
   # torch.ops.load_library("libcupyc.so") # CUDA build

   pt  = torch.tensor([100.0]).double()
   eta = torch.tensor([1.5]).double()
   phi = torch.tensor([0.5]).double()
   e   = torch.tensor([120.0]).double()

   # polar (pT, eta, phi, E) -> Cartesian (px, py, pz, E)
   pmc = torch.ops.tpyc.transform_separate_pxpypze(pt, eta, phi, e)

   # compute invariant mass from Cartesian four-momentum
   m   = torch.ops.tpyc.physics_cartesian_combined_m(pmc)

Languages and Technologies
--------------------------

* **C++20** — core engine, modules, CUDA wrappers.
* **Cython** — Python/C++ bridge with minimal overhead.
* **CUDA** — GPU kernels for physics computations.
* **LibTorch** — tensor operations inside CUDA kernels.
* **Doxygen + Breathe + Sphinx** — documentation pipeline.
