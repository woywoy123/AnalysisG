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
   to populate the output map with heap-allocated instances.  The ``assign_vector``
   helper allocates one typed instance per ROOT-entry row and fills its kinematics::

     // particle header (my_particle.h)
     #include <templates/particle_template.h>
     class Top : public particle_template {
     public:
         Top();
         particle_template* clone() override;
         void build(std::map<std::string, particle_template*>* prt,
                    element_t* el) override;
         int from_res = 0;
     };

     // particle source (my_particle.cxx)
     Top::Top() : particle_template() {
         this->type = "top";
         this->add_leaf("pt",       "_pt");
         this->add_leaf("eta",      "_eta");
         this->add_leaf("phi",      "_phi");
         this->add_leaf("e",        "_e");
         this->add_leaf("index",    "_index");
         this->add_leaf("from_res", "_FromRes");
         this->apply_type_prefix();
     }
     particle_template* Top::clone(){return new Top();}
     void Top::build(std::map<std::string, particle_template*>* prt,
                     element_t* el){
         // Retrieve per-particle branch vectors from the ROOT entry
         std::vector<float> _pt, _eta, _phi, _e;
         std::vector<int>   _index, _from_res;
         el->get("pt",       &_pt);
         el->get("eta",      &_eta);
         el->get("phi",      &_phi);
         el->get("e",        &_e);
         el->get("index",    &_index);
         el->get("from_res", &_from_res);
         for (size_t i = 0; i < _pt.size(); ++i){
             Top* t   = new Top();
             t->pt    = _pt[i];
             t->eta   = _eta[i];
             t->phi   = _phi[i];
             t->e     = _e[i];
             t->index = _index[i];
             t->from_res = _from_res[i];
             (*prt)[std::string(t->hash)] = t;
         }
     }

2. **Define events** — subclass :class:`event_template`, set ``this->name``,
   add ROOT-leaf mappings with ``add_leaf``, register each concrete particle
   class via ``register_particle(&this->m_<collection>)``, set
   ``this->trees``, and override ``build`` / ``CompileEvent``.
   ``build`` reads scalar event quantities; ``CompileEvent`` copies private
   ``std::map`` entries into the public ``std::vector<particle_template*>``
   members and builds any inter-particle associations::

     // event header (my_event.h)
     #include <templates/event_template.h>
     #include "my_particle.h"
     class MyEvent : public event_template {
     public:
         MyEvent();
         event_template* clone() override;
         void build(element_t* el) override;
         void CompileEvent() override;
         std::vector<particle_template*> Tops = {};
         float met = 0;
     private:
         std::map<std::string, Top*> m_tops = {};
     };

     // event source (my_event.cxx)
     MyEvent::MyEvent(){
         this->name = "my_event";
         this->trees = {"nominal"};
         this->add_leaf("met", "met_met");
         this->register_particle(&this->m_tops);
     }
     event_template* MyEvent::clone(){return new MyEvent();}
     void MyEvent::build(element_t* el){el->get("met", &this->met);}
     void MyEvent::CompileEvent(){
         std::map<std::string, Top*>::iterator it;
         for (it = m_tops.begin(); it != m_tops.end(); ++it)
             this->Tops.push_back(it->second);
     }

3. **Define graphs** — subclass :class:`graph_template`, set ``this->name``,
   and override ``CompileEvent``.  Inside ``CompileEvent`` call
   ``get_event<MyEvent>()`` to retrieve the populated event object, pass a
   particle collection to ``define_particle_nodes``, then register feature
   functions.

   Feature functions are plain C++ functions with the signature
   ``void fn(OutputType* out, ParticleOrEventType* in)``; for edge features
   the second argument is ``std::tuple<O*, O*>*``::

     // graph feature functions (my_features.cxx)
     // node feature: pT of each particle
     void node_pt(double* o, particle_template* p){*o = p->pt;}

     // graph-level truth label: 1 if the event contains >=1 resonance top
     void is_signal(bool* o, MyEvent* ev){
         for (size_t i = 0; i < ev->Tops.size(); ++i){
             if (((Top*)ev->Tops[i])->from_res){*o = true; return;}
         }
         *o = false;
     }

     // edge truth label: 1 if both endpoints share the same top index
     void same_top(int* o,
                   std::tuple<particle_template*, particle_template*>* e_ij){
         *o = (std::get<0>(*e_ij)->index == std::get<1>(*e_ij)->index) ? 1 : 0;
     }

     // graph CompileEvent
     void MyGraph::CompileEvent(){
         MyEvent* ev = this->get_event<MyEvent>();
         this->define_particle_nodes(&ev->Tops);

         this->add_graph_truth_feature<bool, MyEvent>(ev, is_signal, "signal");
         this->add_node_data_feature<double, particle_template>(node_pt, "pt");
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

``pyc`` kernels are exposed as PyTorch custom operators registered under the
``tpyc`` (CPU) or ``cupyc`` (CUDA) namespace.  After loading the shared
library the operators are available as ``torch.ops.<ns>.<name>``.
The ``separate`` variants take one tensor per kinematic component; the
``combined`` variants take a single Nx4 tensor.  Both variants return a
2-D tensor (NxK)::

   import torch
   torch.ops.load_library("/path/to/build/libtpyc.so")   # CPU build
   # torch.ops.load_library("/path/to/build/libcupyc.so") # CUDA build

   # Two particles: polar coordinates stacked as Nx4 (pt, eta, phi, E)
   pmu = torch.tensor([[207050.75, 0.562, 2.263, 296197.3],
                        [100000.00, 1.200, 0.500, 115000.0]],
                       dtype=torch.float64)

   # --- separate variant: pass each column individually ---
   pmc_sep = torch.ops.tpyc.transform_separate_pxpypze(
       pmu[:, 0],   # pt
       pmu[:, 1],   # eta
       pmu[:, 2],   # phi
       pmu[:, 3],   # E
   )   # returns Nx4: (px, py, pz, E)

   # --- combined variant: pass the Nx4 polar tensor directly ---
   pmc_comb = torch.ops.tpyc.transform_combined_pxpypze(pmu)

   # Invariant mass from Cartesian four-momentum (Nx4 input -> Nx1 output)
   mass = torch.ops.tpyc.physics_cartesian_combined_m(pmc_comb)

Languages and Technologies
--------------------------

* **C++20** — core engine, modules, CUDA wrappers.
* **Cython** — Python/C++ bridge with minimal overhead.
* **CUDA** — GPU kernels for physics computations.
* **LibTorch** — tensor operations inside CUDA kernels.
* **Doxygen + Breathe + Sphinx** — documentation pipeline.
