Quick Start
===========

A complete AnalysisG analysis follows this workflow:

1. Write C++ particle, event, and graph classes.
2. Write thin Cython (`.pxd` / `.pyx`) wrappers to expose them to Python.
3. Compile with ``scikit-build-core`` (``pip install .``).
4. Import the compiled Cython classes and run the pipeline via
   the Python :class:`Analysis` class.

All examples below reflect the patterns used in the
`bsm_4tops <https://github.com/woywoy123/AnalysisG/tree/master/src/AnalysisG/events/bsm_4tops>`_
and
`grift <https://github.com/woywoy123/AnalysisG/tree/master/src/AnalysisG/models/grift>`_
reference implementations included in the repository.

Step 1 — Define Particles
--------------------------

Subclass :cpp:class:`particle_template`, set ``this->type``, register ROOT
branch names with :cpp:func:`particle_template::add_leaf`, and override
:cpp:func:`particle_template::build` to allocate typed instances from a ROOT
entry.

.. note::
   The second argument of ``add_leaf(key, suffix)`` is an *underscore-prefixed
   suffix* of the ROOT branch name.  Calling :cpp:func:`particle_template::apply_type_prefix`
   afterwards prepends ``this->type`` to every suffix, so ``add_leaf("pt", "_pt")``
   with ``this->type = "top"`` resolves to the ROOT branch ``"top_pt"``.
   Do **not** pre-include the type prefix in the suffix: that would create a
   doubled prefix (e.g. ``"toptop_pt"``).

**Header** (``my_particle.h``):

.. code-block:: cpp

   #include <templates/particle_template.h>

   class Top : public particle_template {
   public:
       Top();
       ~Top();
       particle_template* clone() override;  // Required factory method
       void build(std::map<std::string, particle_template*>* prt,
                  element_t* el) override;
       int from_res = 0;
       int status   = 0;
   };

.. note::
   ``assign_vector`` is a **user-defined** template helper (not part of the
   core framework).  It is defined in
   ``bsm_4tops/include/bsm_4tops/particles.h`` as a convenience for filling
   the standard kinematic fields from ``element_t``.  You can copy it from
   that reference implementation or write the equivalent loop directly using
   ``element_t::get()``.

**Source** (``my_particle.cxx``):

.. code-block:: cpp

   #include "my_particle.h"

   Top::Top() : particle_template() {
       this->type = "top";
       // Second argument is the ROOT branch suffix; apply_type_prefix()
       // prepends this->type, so "_pt" resolves to the branch "top_pt".
       this->add_leaf("pt",       "_pt");
       this->add_leaf("eta",      "_eta");
       this->add_leaf("phi",      "_phi");
       this->add_leaf("e",        "_e");
       this->add_leaf("index",    "_index");
       this->add_leaf("pdgid",    "_pdgid");
       this->add_leaf("from_res", "_FromRes");
       this->add_leaf("status",   "_status");
       this->apply_type_prefix();   // leaves now map to "top_pt", "top_eta", …
   }

   Top::~Top() {}
   particle_template* Top::clone() { return (particle_template*)new Top(); }

   void Top::build(std::map<std::string, particle_template*>* prt,
                   element_t* el) {
       // Use element_t::get() to read branch data directly (no helper needed)
       std::vector<float> _pt, _eta, _phi, _e;
       std::vector<int>   _index, _pdgid, _from_res, _status;
       el->get("pt",       &_pt);
       el->get("eta",      &_eta);
       el->get("phi",      &_phi);
       el->get("e",        &_e);
       el->get("index",    &_index);
       el->get("pdgid",    &_pdgid);
       el->get("from_res", &_from_res);
       el->get("status",   &_status);

       for (int x = 0; x < (int)_pt.size(); ++x) {
           Top* t    = new Top();
           t->pt     = _pt[x];   t->eta = _eta[x];
           t->phi    = _phi[x];  t->e   = _e[x];
           t->index  = _index[x]; t->pdgid = _pdgid[x];
           t->from_res = _from_res[x];
           t->status   = _status[x];
           (*prt)[std::string(t->hash)] = t;
       }
   }

Step 2 — Define an Event
-------------------------

Subclass :cpp:class:`event_template`, declare the particle maps, set
``this->trees`` (ROOT tree names), register scalar branch names with
``add_leaf``, register particle maps with
:cpp:func:`event_template::register_particle`, and override
:cpp:func:`event_template::build` / :cpp:func:`event_template::CompileEvent`.

.. note::
   ``sort_by_index`` and ``vectorize`` are **user-defined** private template
   methods (not in ``event_template``).  They are defined in the header below,
   following the pattern in ``bsm_4tops/include/bsm_4tops/event.h``.

**Header** (``my_event.h``):

.. code-block:: cpp

   #include <templates/event_template.h>
   #include "my_particle.h"

   class MyEvent : public event_template {
   public:
       MyEvent();
       ~MyEvent();
       event_template* clone() override;  // Required factory method
       void build(element_t* el) override;
       void CompileEvent() override;

       std::vector<particle_template*> Tops = {};
       float met = 0;
   private:
       std::map<std::string, Top*> m_tops = {};

       // User-defined utility: sort particles by their index field
       template <typename G>
       std::map<int, G*> sort_by_index(std::map<std::string, G*>* ipt) {
           std::map<int, G*> data;
           for (auto ix = ipt->begin(); ix != ipt->end(); ++ix)
               data[int(ix->second->index)] = ix->second;
           return data;
       }

       // User-defined utility: flatten index-keyed map into a vector
       template <typename m, typename G>
       void vectorize(std::map<m, G*>* ipt, std::vector<particle_template*>* vec) {
           for (auto ix = ipt->begin(); ix != ipt->end(); ++ix)
               vec->push_back(ix->second);
       }
   };

**Source** (``my_event.cxx``):

.. code-block:: cpp

   #include "my_event.h"

   MyEvent::MyEvent() {
       this->name  = "my_event";
       this->trees = {"nominal"};
       this->add_leaf("met", "met_met");
       this->register_particle(&this->m_tops);
   }

   MyEvent::~MyEvent() {}
   event_template* MyEvent::clone() { return new MyEvent(); }

   void MyEvent::build(element_t* el) {
       el->get("met", &this->met);
   }

   void MyEvent::CompileEvent() {
       // Sort into index-keyed map then vectorise
       std::map<int, Top*> sorted = this->sort_by_index(&this->m_tops);
       this->vectorize(&sorted, &this->Tops);
   }

Step 3 — Define a Graph
------------------------

Subclass :cpp:class:`graph_template` and override
:cpp:func:`graph_template::CompileEvent`.  Inside ``CompileEvent``:

* Retrieve the typed event with :cpp:func:`graph_template::get_event`.
* Pass a particle vector to :cpp:func:`graph_template::define_particle_nodes`
  to set up nodes (one node per particle).
* Register feature functions via the ``add_*_feature`` family:

  * ``add_node_data_feature<T, P>(fn, name)`` — per-node input feature
  * ``add_node_truth_feature<T, P>(fn, name)`` — per-node truth label
  * ``add_edge_data_feature<T, P>(fn, name)`` — per-edge input feature
  * ``add_edge_truth_feature<T, P>(fn, name)`` — per-edge truth label
  * ``add_graph_data_feature<T, Ev>(ev, fn, name)`` — graph-level input
  * ``add_graph_truth_feature<T, Ev>(ev, fn, name)`` — graph-level truth

Feature functions are free (non-member) functions with signatures:

* Node/graph features: ``void fn(OutputType* out, InputType* in)``
* Edge features: ``void fn(OutputType* out, std::tuple<P*, P*>* edge)``

**Graph source** (``my_graph.cxx``):

.. code-block:: cpp

   #include "my_graph.h"   // inherits graph_template
   #include "my_event.h"
   #include "my_particle.h"

   // --- feature functions ---

   // node feature: transverse momentum
   void node_pt(double* out, particle_template* p) { *out = p->pt; }

   // graph-level truth: 1 if ≥1 top is from a resonance
   void signal_event(bool* out, MyEvent* ev) {
       *out = false;
       for (auto* p : ev->Tops) {
           if (static_cast<Top*>(p)->from_res) { *out = true; return; }
       }
   }

   // edge truth: 1 if both endpoints share the same top-quark index
   void top_edge(int* out, std::tuple<particle_template*, particle_template*>* e) {
       *out = (std::get<0>(*e)->index == std::get<1>(*e)->index) ? 1 : 0;
   }

   // --- CompileEvent ---
   void MyGraph::CompileEvent() {
       MyEvent* ev = this->get_event<MyEvent>();
       this->define_particle_nodes(&ev->Tops);

       this->add_graph_truth_feature<bool, MyEvent>(ev, signal_event, "signal");
       this->add_node_data_feature<double, particle_template>(node_pt, "pt");
       this->add_edge_truth_feature<int,   particle_template>(top_edge, "top_edge");
   }

Step 4 — Cython Interfaces
----------------------------

Every C++ class passed to the Python :class:`Analysis` API must be wrapped in
a thin Cython layer.  Particles are constructed internally by the framework
and do not need a Python-facing wrapper — only events, graphs, models, and
selections do.

**Event** (``my_event.pxd`` + ``my_event.pyx``):

.. code-block:: cython

   # my_event.pxd
   # distutils: language=c++
   # cython: language_level=3

   from AnalysisG.core.event_template cimport event_template, EventTemplate

   cdef extern from "<my_module/my_event.h>":
       cdef cppclass MyEventCpp(event_template):
           MyEventCpp() except+

   cdef class PyMyEvent(EventTemplate):
       cdef MyEventCpp* tt

.. code-block:: cython

   # my_event.pyx
   # distutils: language=c++
   # cython: language_level=3

   from AnalysisG.core.event_template cimport EventTemplate
   from my_event cimport MyEventCpp

   cdef class PyMyEvent(EventTemplate):
       def __cinit__(self):
           self.tt  = new MyEventCpp()
           self.ptr = <event_template*>(self.tt)   # cast to base pointer
       def __init__(self): pass
       def __dealloc__(self): del self.tt

**Graph** (``my_graph.pxd`` + ``my_graph.pyx``):

.. code-block:: cython

   # my_graph.pxd
   # distutils: language=c++
   # cython: language_level=3

   from AnalysisG.core.graph_template cimport graph_template, GraphTemplate

   cdef extern from "<my_module/my_graph.h>":
       cdef cppclass MyGraphCpp(graph_template):
           MyGraphCpp() except+

   cdef class PyMyGraph(GraphTemplate): pass

.. code-block:: cython

   # my_graph.pyx
   # distutils: language=c++
   # cython: language_level=3

   from AnalysisG.core.graph_template cimport GraphTemplate
   from my_graph cimport MyGraphCpp

   cdef class PyMyGraph(GraphTemplate):
       def __cinit__(self): self.ptr = new MyGraphCpp()
       def __init__(self): pass
       def __dealloc__(self): del self.ptr

**Model** (``my_model.pxd`` + ``my_model.pyx``):

.. code-block:: cython

   # my_model.pxd
   # distutils: language=c++
   # cython: language_level=3

   from AnalysisG.core.model_template cimport model_template, ModelTemplate

   cdef extern from "<my_module/my_model.h>":
       cdef cppclass MyModelCpp(model_template):
           MyModelCpp() except+

   cdef class PyMyModel(ModelTemplate): pass

.. code-block:: cython

   # my_model.pyx
   # distutils: language=c++
   # cython: language_level=3

   from AnalysisG.core.model_template cimport ModelTemplate
   from my_model cimport MyModelCpp

   cdef class PyMyModel(ModelTemplate):
       def __cinit__(self): self.nn_ptr = new MyModelCpp()
       def __init__(self): pass
       def __dealloc__(self): del self.nn_ptr

Step 5 — Define a Model *(optional)*
--------------------------------------

Subclass :cpp:class:`model_template` and override
:cpp:func:`model_template::forward`.  Inside ``forward``, fetch tensors from
the ``graph_t`` object with ``data->get_data_*(name, this)`` (the second
argument is always ``this`` — the model pointer used to select the correct
device) and write predictions with ``prediction_*_feature(name, tensor)``.

Register PyTorch sub-modules with
:cpp:func:`model_template::register_module` in the constructor.

**Header** (``my_model.h``):

.. code-block:: cpp

   #include <templates/model_template.h>

   class MyModel : public model_template {
   public:
       MyModel();
       ~MyModel();
       model_template* clone() override;
       void forward(graph_t* data) override;

       torch::nn::Sequential* node_mlp = nullptr;
   };

**Source** (``my_model.cxx``):

.. code-block:: cpp

   #include "my_model.h"

   MyModel::MyModel() {
       this->node_mlp = new torch::nn::Sequential({
           {"l1", torch::nn::Linear(4, 64)},
           {"r1", torch::nn::ReLU()},
           {"l2", torch::nn::Linear(64, 1)}
       });
       this->register_module(this->node_mlp);
   }

   MyModel::~MyModel() {}
   model_template* MyModel::clone() { return new MyModel(); }

   void MyModel::forward(graph_t* data) {
       // Fetch input tensors — second arg is always `this` (selects device)
       torch::Tensor node_pt  = data->get_data_node("pt",    this)->clone();
       torch::Tensor node_eta = data->get_data_node("eta",   this)->clone();
       torch::Tensor met      = data->get_data_graph("met",  this)->clone();
       torch::Tensor edge_idx = data->get_edge_index(this)->to(torch::kLong);

       torch::Tensor feats = torch::cat(
           {node_pt, node_eta, met.expand_as(node_pt), met.expand_as(node_pt)}, -1);
       torch::Tensor out = (*node_mlp)->forward(feats);

       // Write node-level prediction
       this->prediction_node_feature("top_node", out);

       // Extra output only written during inference
       if (!this->inference_mode) { return; }
       this->prediction_extra("node_score", torch::sigmoid(out));
   }

Step 6 — Run the Pipeline (Python)
-------------------------------------

After compiling with ``pip install .``, import the generated Cython classes
and wire them together via the Python :class:`Analysis` class.  All property
values shown below are verified defaults or typical overrides.

.. code-block:: python

   from AnalysisG import Analysis
   from AnalysisG.core.lossfx import OptimizerConfig
   from my_module import PyMyEvent, PyMyGraph, PyMyModel  # compiled Cython classes

   # --- configure optimizer ---
   op = OptimizerConfig()
   op.Optimizer    = "Adam"   # "Adam", "SGD", "RMSprop", "Adagrad", "LBFGS"
   op.Scheduler    = "StepLR" # "StepLR", "CyclicLR", "ExponentialLR"
   op.lr           = 1e-3
   op.weight_decay = 1e-4
   op.step_size    = 10       # epochs between LR decay steps
   op.gamma        = 0.1      # multiplicative decay factor

   # --- build pipeline ---
   ana = Analysis()
   ana.OutputPath  = "./output"    # default: './ProjectName'
   ana.Epochs      = 20            # default: 10
   ana.kFolds      = 5             # default: 10  (number of folds)
   ana.kFold       = []            # default: []  ([] = run all folds)
   ana.TrainSize   = 80.0          # percentage 0–100 (default: 50.0)
   ana.BatchSize   = 32            # default: 1
   ana.Threads     = 4             # default: 10
   ana.BuildCache  = True          # build the graph HDF5 cache
   ana.Training    = True          # enable training phase (default: True)
   ana.Validation  = True          # enable validation phase (default: True)
   ana.Evaluation  = True          # enable evaluation phase (default: True)

   ana.AddSamples("./data/ttbar.root", "ttbar")
   ana.AddEvent(PyMyEvent(), "ttbar")
   ana.AddGraph(PyMyGraph(), "ttbar")
   ana.AddModel(PyMyModel(), op, "run1")

   ana.Start()

Step 7 — Define Selections *(optional)*
-----------------------------------------

Subclass :cpp:class:`selection_template` and implement
:cpp:func:`selection_template::selection` for per-event logic.  Optional
overrides include:

* ``strategy()`` — aggregate post-processing over all passed events
* ``Postprocessing()`` — finalise and serialise results
* ``InterpretROOT()`` — re-read results from a ROOT output file
* ``dump()`` / ``load()`` — pickle serialisation

Counting Jets with IO
-----------------------

The :class:`IO` class can be used independently to inspect ROOT files without
running the full pipeline.  Keys in the yielded dict are ``bytes`` in the
format ``b'tree.branch.leaf'``.

.. code-block:: python

   from AnalysisG.core.io import IO

   reader = IO()
   reader.Files  = ["test/samples/dilepton/DAOD_TOPQ1.21955717._000001.root"]
   reader.Trees  = ["nominal"]
   reader.Leaves = ["jet_pt"]
   reader.ScanKeys()          # build the internal key index before iterating

   n_events = 0
   n_jets   = 0
   for entry in reader:
       key = b'nominal.jet_pt.jet_pt'
       if key not in entry:
           continue
       n_events += 1
       n_jets   += len(entry[key])

   print(f"Events : {n_events}")   # 1098
   print(f"Jets   : {n_jets}")     # 8161
   print(f"Avg    : {n_jets / n_events:.2f} jets/event")  # 7.43

.. note::
   **Verified output** from the dilepton test sample
   (``DAOD_TOPQ1.21955717._000001.root``, tree ``nominal``):

   * Events: **1,098**
   * Total jets (``b'nominal.jet_pt.jet_pt'``): **8,161**
   * Average jets per event: **7.43** (min 4, max 14)

Using pyc CUDA Kernels
-----------------------

The ``pyc`` sub-package registers HEP-specific kernels as PyTorch custom
operators under the ``tpyc`` (CPU) or ``cupyc`` (CUDA) namespaces.
After building the project the shared library is located at
``<build_dir>/pyc/interface/lib{tpyc|cupyc}.so``.

.. code-block:: python

   import torch

   # Load the shared library (path relative to your build directory)
   torch.ops.load_library("build/pyc/interface/libtpyc.so")   # CPU
   # torch.ops.load_library("build/pyc/interface/libcupyc.so") # CUDA

   # Nx4 tensor in polar coordinates (pt, eta, phi, E) — float64
   pmu = torch.tensor(
       [[207050.75, 0.562, 2.263, 296197.3],
        [100000.00, 1.200, 0.500, 115000.0]],
       dtype=torch.float64
   )

   # Convert cylindrical (pt, eta, phi, E) → Cartesian (px, py, pz, E) — Nx4 result
   pmc = torch.ops.tpyc.transform_combined_pxpypze(pmu)

   # Invariant mass from Cartesian four-momentum — Nx1 result
   mass = torch.ops.tpyc.physics_cartesian_combined_m(pmc)
