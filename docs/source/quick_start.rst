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
       particle_template* clone() override;  // Returns a new heap-allocated Top instance
       void build(std::map<std::string, particle_template*>* prt,
                  element_t* el) override;
       int from_res = 0;
   };

**Source** (``my_particle.cxx``):

.. code-block:: cpp

   #include "my_particle.h"

   Top::Top() : particle_template() {
       this->type = "top";
       // Second argument is the ROOT branch suffix; apply_type_prefix()
       // prepends this->type so "_pt" becomes the branch "top_pt", etc.
       this->add_leaf("pt",       "_pt");
       this->add_leaf("eta",      "_eta");
       this->add_leaf("phi",      "_phi");
       this->add_leaf("e",        "_e");
       this->add_leaf("index",    "_index");
       this->add_leaf("from_res", "_FromRes");
       this->apply_type_prefix();   // leaves now map to "top_pt", "top_eta", ...
   }

   particle_template* Top::clone() { return new Top(); }  // Required factory method

   void Top::build(std::map<std::string, particle_template*>* prt,
                   element_t* el) {
       std::vector<float> _pt, _eta, _phi, _e;
       std::vector<int>   _index, _from_res;
       el->get("pt",       &_pt);
       el->get("eta",      &_eta);
       el->get("phi",      &_phi);
       el->get("e",        &_e);
       el->get("index",    &_index);
       el->get("from_res", &_from_res);
       for (size_t i = 0; i < _pt.size(); ++i) {
           Top* t    = new Top();
           t->pt     = _pt[i];
           t->eta    = _eta[i];
           t->phi    = _phi[i];
           t->e      = _e[i];
           t->index  = _index[i];
           t->from_res = _from_res[i];
           (*prt)[std::string(t->hash)] = t;
       }
   }

Step 2 — Define an Event
-------------------------

Subclass :cpp:class:`event_template`, declare the particle collections, set
``this->trees`` (ROOT tree names), register particle maps with
:cpp:func:`event_template::register_particle`, and override
:cpp:func:`event_template::build` / :cpp:func:`event_template::CompileEvent`.

**Header** (``my_event.h``):

.. code-block:: cpp

   #include <templates/event_template.h>
   #include "my_particle.h"

   class MyEvent : public event_template {
   public:
       MyEvent();
       event_template* clone() override;  // Required factory method for event creation
       void build(element_t* el) override;
       void CompileEvent() override;

       std::vector<particle_template*> Tops = {};
       float met = 0;
   private:
       std::map<std::string, Top*> m_tops = {};
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

   event_template* MyEvent::clone() { return new MyEvent(); }  // Required factory method

   void MyEvent::build(element_t* el) {
       el->get("met", &this->met);
   }

   void MyEvent::CompileEvent() {
       for (auto& [key, top] : m_tops)  // extract particle pointers from hash-keyed map
           this->Tops.push_back(top);
   }

Step 3 — Define a Graph
------------------------

Subclass :cpp:class:`graph_template`, override
:cpp:func:`graph_template::CompileEvent`.  Inside ``CompileEvent``:

* Retrieve the event with :cpp:func:`graph_template::get_event`.
* Pass a particle collection to :cpp:func:`graph_template::define_particle_nodes`.
* Register feature functions via the ``add_*_feature`` family.

Feature functions are plain C++ free functions:

* **Node/graph features**: ``void fn(OutputType* out, InputType* in)``
* **Edge features**: ``void fn(OutputType* out, std::tuple<P*, P*>* edge)``

**Graph source** (``my_graph.cxx``):

.. code-block:: cpp

   #include "my_graph.h"   // inherits graph_template
   #include "my_event.h"
   #include "my_particle.h"

   // --- feature functions (free functions, not class members) ---

   // node feature: transverse momentum of each particle
   void node_pt(double* out, particle_template* p) { *out = p->pt; }

   // graph-level truth label: 1 if ≥1 top quark is from a resonance
   void is_signal(bool* out, MyEvent* ev) {
       for (auto* p : ev->Tops) {
           if (static_cast<Top*>(p)->from_res) { *out = true; return; }
       }
       *out = false;
   }

   // edge truth label: 1 if both endpoints share the same top-quark index
   void same_top(int* out, std::tuple<particle_template*, particle_template*>* e) {
       *out = (std::get<0>(*e)->index == std::get<1>(*e)->index) ? 1 : 0;
   }

   // --- CompileEvent ---
   void MyGraph::CompileEvent() {
       MyEvent* ev = this->get_event<MyEvent>();
       this->define_particle_nodes(&ev->Tops);

       this->add_graph_truth_feature<bool, MyEvent>(ev, is_signal, "signal");
       this->add_node_data_feature<double, particle_template>(node_pt, "pt");
       this->add_edge_truth_feature<int, particle_template>(same_top, "same_top");
   }

Step 4 — Define a Model *(optional)*
--------------------------------------

Subclass :cpp:class:`model_template` and override
:cpp:func:`model_template::forward`.  Inside ``forward``, fetch tensors
from the ``graph_t`` object with ``data->get_data_*(name, this)`` (the
second argument is always ``this`` — the model pointer used to select the
correct device tensor) and write predictions with
``prediction_*_feature(name, tensor)``.

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
       torch::Tensor node_pt  = data->get_data_node("pt",  this)->clone();
       torch::Tensor node_eta = data->get_data_node("eta", this)->clone();
       torch::Tensor met      = data->get_data_graph("met", this)->clone();
       torch::Tensor edge_idx = data->get_edge_index(this)->to(torch::kLong);

       torch::Tensor feats = torch::cat({node_pt, node_eta, met.expand_as(node_pt),
                                         met.expand_as(node_pt)}, -1);
       torch::Tensor out = (*node_mlp)->forward(feats);

       // Write predictions back to the graph object
       this->prediction_node_feature("top_node", out);

       // prediction_extra is only written during inference (not training)
       if (!this->inference_mode) { return; }
       this->prediction_extra("node_score", torch::sigmoid(out));
   }

Step 5 — Define Selections *(optional)*
-----------------------------------------

Subclass :cpp:class:`selection_template` and implement
:cpp:func:`selection_template::selection` and optionally
:cpp:func:`selection_template::strategy` for per-event logic and aggregate
post-processing.  See ``selectiontemplate.md`` in the templates directory for
a complete C++ + Cython example.

Step 6 — Cython Interfaces
----------------------------

Every C++ class that is passed to the Python :class:`Analysis` API must be
wrapped in a thin Cython layer.  Particles are built internally by the
framework and do not need a Python-facing Cython class — only events, graphs,
models, and selections do.

**Event** (``my_event.pxd`` + ``my_event.pyx``):

.. code-block:: cython

   # my_event.pxd
   # distutils: language=c++
   # cython: language_level=3

   from AnalysisG.core.event_template cimport event_template, EventTemplate

   cdef extern from "<my_module/my_event.h>":
       cdef cppclass MyEvent(event_template):
           MyEvent() except+

   cdef class PyMyEvent(EventTemplate):
       cdef MyEvent* tt

.. code-block:: cython

   # my_event.pyx
   # distutils: language=c++
   # cython: language_level=3

   from AnalysisG.core.event_template cimport EventTemplate
   from my_event cimport MyEvent

   cdef class PyMyEvent(EventTemplate):
       def __cinit__(self):
           self.tt  = new MyEvent()
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
       cdef cppclass MyModel(model_template):
           MyModel() except+

   cdef class PyMyModel(ModelTemplate): pass

.. code-block:: cython

   # my_model.pyx
   # distutils: language=c++
   # cython: language_level=3

   from AnalysisG.core.model_template cimport ModelTemplate
   from my_model cimport MyModel

   cdef class PyMyModel(ModelTemplate):
       def __cinit__(self): self.nn_ptr = new MyModel()
       def __init__(self): pass
       def __dealloc__(self): del self.nn_ptr

Step 7 — Run the Pipeline (Python)
-------------------------------------

After compiling with ``pip install .``, import the generated Cython classes
and wire them together via the Python :class:`Analysis` class:

.. code-block:: python

   from AnalysisG import Analysis
   from AnalysisG.core.lossfx import OptimizerConfig
   from my_module import PyMyEvent, PyMyGraph, PyMyModel  # compiled Cython classes

   # --- configure optimiser ---
   op = OptimizerConfig()
   op.Optimizer = "adam"
   op.lr = 1e-3

   # --- build pipeline ---
   ana = Analysis()
   ana.OutputPath  = "./output"
   ana.Epochs      = 20
   ana.kFolds      = 10
   ana.TrainSize   = 80   # percentage: 80 % of graphs used for training
   ana.Threads     = 4

   ana.AddSamples("./data/ttbar.root", "ttbar")
   ana.AddEvent(PyMyEvent(), "ttbar")
   ana.AddGraph(PyMyGraph(), "ttbar")
   ana.AddModel(PyMyModel(), op, "run1")

   ana.Start()

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
