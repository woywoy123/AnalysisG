Quick Start
===========

A typical AnalysisG analysis follows five steps:
define particles → define an event → define a graph → (optionally) define
selections and models → run the pipeline via the Python :class:`Analysis` class.

All examples below are derived from the
`bsm_4tops <https://github.com/woywoy123/AnalysisG/tree/master/truth-studies/bsm_4tops>`_
reference analysis included in the repository.

Step 1 — Define Particles
--------------------------

Subclass :cpp:class:`particle_template`, set ``this->type``, register ROOT
branch names with :cpp:func:`particle_template::add_leaf`, and override
:cpp:func:`particle_template::build` to allocate typed instances from a ROOT
entry.

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
       this->add_leaf("pt",       "top_pt");
       this->add_leaf("eta",      "top_eta");
       this->add_leaf("phi",      "top_phi");
       this->add_leaf("e",        "top_e");
       this->add_leaf("index",    "top_index");
       this->add_leaf("from_res", "top_FromRes");
       this->apply_type_prefix();
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

Step 4 — Define Selections *(optional)*
-----------------------------------------

Subclass :cpp:class:`selection_template` and implement
:cpp:func:`selection_template::selection` and optionally
:cpp:func:`selection_template::strategy` for per-event logic and aggregate
post-processing.

Step 5 — Run the Pipeline (Python)
------------------------------------

Wire all user-defined components together via the Python :class:`Analysis`
class:

.. code-block:: python

   from AnalysisG import Analysis
   from AnalysisG.core.lossfx import OptimizerConfig

   # --- optional: configure optimiser ---
   op = OptimizerConfig()
   op.Optimizer = "adam"
   op.lr = 1e-3

   # --- build pipeline ---
   ana = Analysis()
   ana.OutputPath  = "./output"
   ana.Epochs      = 20
   ana.kFolds      = 10
   ana.TrainSize   = 0.8
   ana.Threads     = 4

   ana.AddSamples("./data/ttbar.root", "ttbar")
   ana.AddEvent(MyEvent(),   "ttbar")
   ana.AddGraph(MyGraph(),   "ttbar")
   ana.AddModel(MyModel(), op, "run1")

   ana.Start()

Using pyc CUDA Kernels
-----------------------

The ``pyc`` sub-package registers HEP-specific kernels as PyTorch custom
operators under the ``tpyc`` (CPU) or ``cupyc`` (CUDA) namespaces.

.. code-block:: python

   import torch

   # Load the shared library built alongside the package
   torch.ops.load_library("/path/to/build/libtpyc.so")   # CPU
   # torch.ops.load_library("/path/to/build/libcupyc.so") # CUDA

   # Nx4 tensor in polar coordinates (pt, eta, phi, E) — float64
   pmu = torch.tensor(
       [[207050.75, 0.562, 2.263, 296197.3],
        [100000.00, 1.200, 0.500, 115000.0]],
       dtype=torch.float64
   )

   # Convert to Cartesian (px, py, pz, E) — Nx4 result
   pmc = torch.ops.tpyc.transform_combined_pxpypze(pmu)

   # Invariant mass from Cartesian four-momentum — Nx1 result
   mass = torch.ops.tpyc.physics_cartesian_combined_m(pmc)
