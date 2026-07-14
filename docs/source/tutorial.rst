End-to-End Pipeline Tutorial
============================

AnalysisG is designed around a single, fundamental philosophy: **The Event is the source of truth.** 

Once you have defined your physics particles and mapped them into an Event from a ROOT file, that exact same Event can be split down three distinct downstream paths without rewriting any data-loading logic:
1. **Graph Compilation**: Transforming the event into a graph structure for Neural Network training.
2. **Selection Analysis**: Running cut-based analyses and aggregating kinematics.
3. **Inference**: Consuming pre-trained model predictions to reconstruct topologies (like Top quarks).

This tutorial walks through building a full pipeline from start to finish.

1. The Fundamental Building Blocks
----------------------------------

Before we can split our workflows, we must define the atomic units of our physics: the Particle and the Event.

Step 1a: Defining the Particle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A Particle inherits from :cpp:class:`particle_template`. We register the ROOT branch suffixes it expects. The framework will automatically handle reading these branches from the `.root` file.

.. code-block:: cpp

   #include <templates/particle_template.h>

   class Top : public particle_template {
   public:
       Top() {
           this->type = "top";
           this->add_leaf("pt",  "_pt");
           this->add_leaf("eta", "_eta");
           this->add_leaf("phi", "_phi");
           this->add_leaf("e",   "_e");
           this->apply_type_prefix(); // Maps to: "top_pt", "top_eta", etc.
       }
       ~Top() {}
       particle_template* clone() override { return new Top(); }

       void build(std::map<std::string, particle_template*>* prt, element_t* el) override {
           std::vector<float> _pt, _eta, _phi, _e;
           el->get("pt", &_pt); el->get("eta", &_eta);
           el->get("phi", &_phi); el->get("e", &_e);

           for (size_t x = 0; x < _pt.size(); ++x) {
               Top* t = new Top();
               t->pt = _pt[x]; t->eta = _eta[x]; 
               t->phi = _phi[x]; t->e = _e[x];
               (*prt)[std::string(t->hash)] = t;
           }
       }
   };

Step 1b: Defining the Event
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Event inherits from :cpp:class:`event_template`. It registers our ``Top`` particles so the framework knows to build them when iterating over the ROOT trees.

.. code-block:: cpp

   #include <templates/event_template.h>

   class MyEvent : public event_template {
   public:
       std::vector<particle_template*> Tops = {};
       float met = 0;

       MyEvent() {
           this->name = "my_event";
           this->trees = {"nominal"};
           this->add_leaf("met", "met_met");
           this->register_particle(&this->m_tops); // Register the particle map
       }
       ~MyEvent() {}
       event_template* clone() override { return new MyEvent(); }

       void build(element_t* el) override {
           el->get("met", &this->met);
       }

       void CompileEvent() override {
           // Flatten the internal map into our public Tops vector
           for (auto ix = this->m_tops.begin(); ix != this->m_tops.end(); ++ix) {
               this->Tops.push_back(ix->second);
           }
       }
   private:
       std::map<std::string, Top*> m_tops = {};
   };

2. The Branching Paths (The "Splits")
-------------------------------------

With ``MyEvent`` defined, we have our source of truth. We can now plug this exact event into three different systems.

Split A: Event $\rightarrow$ Graph (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To train a Graph Neural Network, we transform the Event into a Graph by inheriting from :cpp:class:`graph_template`.

.. code-block:: cpp

   #include <templates/graph_template.h>

   // A simple node feature function extracting pT
   void node_pt(double* out, particle_template* p) { *out = p->pt; }

   class MyGraph : public graph_template {
   public:
       MyGraph() { this->name = "my_graph"; }
       ~MyGraph() {}
       graph_template* clone() override { return new MyGraph(); }

       void CompileEvent() override {
           MyEvent* ev = this->get_event<MyEvent>(); // Retrieve our compiled event!
           
           // Register nodes based on the Tops found in the event
           this->define_particle_nodes(&ev->Tops);

           // Attach the pT feature to every node
           this->add_node_data_feature<double, particle_template>(node_pt, "pt");
       }
   };

Split B: Event $\rightarrow$ Selection (Analysis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To perform standard cut-based physics analysis, we inherit from :cpp:class:`selection_template`. This allows us to aggregate properties across millions of events safely.

.. code-block:: cpp

   #include <templates/selection_template.h>

   class TopSelection : public selection_template {
   public:
       TopSelection() { this->name = "high_pt_tops"; }
       ~TopSelection() {}
       selection_template* clone() override { return new TopSelection(); }

       bool selection(event_template* ev) override {
           MyEvent* event = (MyEvent*)ev; // Retrieve our event!

           int high_pt_count = 0;
           for (auto* p : event->Tops) {
               if (p->pt > 500000) { // 500 GeV cut
                   high_pt_count++;
               }
           }
           
           // Reject event if it has no high pT tops
           if (high_pt_count == 0) return false;

           // Save the count for this event. 
           // The framework automatically maps this variable to the event's hash.
           this->write(&high_pt_count, "n_high_pt_tops");
           return true; 
       }
   };

Split C: Event $\rightarrow$ Inference (Reconstruction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
What happens after your GNN is trained? You use a specialised Event (like the built-in ``gnn_event``) to read the neural network outputs directly from an evaluation dataset, and run algorithms (like PageRank) to cluster the nodes back into physical objects.

.. code-block:: cpp

   // Snippet reflecting the logic inside src/AnalysisG/events/gnn/cxx/event.cxx
   void gnn_event::CompileEvent() {
       // 1. We extract the edge probabilities predicted by the GNN
       el->get("top_edge", &this->edge_top_scores);
       
       // 2. We use these probabilities as adjacency weights to cluster particles 
       //    together into Top Quarks using PageRank!
       std::map<int, std::map<int, float>> w_nrm_top;
       // ... populate w_nrm_top with edge_top_scores ...

       // 3. Build reconstructed Top objects based on the unmasked PageRank algorithm
       this->build_particles(&nrm_tops, &w_nrm_top, &this->m_tops, true, pagerank_e::unmasked);
   }

3. Tying It Together (Python API)
---------------------------------

Once your C++ classes are wrapped in Cython (see Quick Start), you manage the entire pipeline from Python using the ``Analysis()`` object. You simply attach the classes you want to run!

.. code-block:: python

   from AnalysisG import Analysis
   from my_module import PyMyEvent, PyMyGraph, PyTopSelection

   ana = Analysis()
   ana.OutputPath = "./output"
   ana.AddSamples("./data/ttbar.root", "ttbar")

   # The Source of Truth
   ana.AddEvent(PyMyEvent(), "ttbar")

   # Split A: Generate Training Graphs
   ana.AddGraph(PyMyGraph(), "ttbar")

   # Split B: Run Cut-Based Analysis
   ana.AddSelection(PyTopSelection())

   # Run everything concurrently!
   ana.Start()

4. Plotting a Simple Distribution
---------------------------------

Once the ``Analysis()`` pipeline completes, the results of your Selection are safely aggregated in your output directories. You can fetch this data and plot it using Python.

.. code-block:: python

   from AnalysisG.core.io import IO
   from AnalysisG.core.plotting import TH1F

   # We read the output file generated by our TopSelection
   reader = IO()
   reader.Files  = ["./output/ttbar/Selections/high_pt_tops.root"]
   reader.Trees  = ["high_pt_tops"]
   reader.Leaves = ["n_high_pt_tops"]
   reader.ScanKeys()

   counts = []
   for entry in reader:
       # The key format is usually: tree_name.branch_name.leaf_name
       key = b'high_pt_tops.n_high_pt_tops.n_high_pt_tops'
       if key in entry:
           counts.append(entry[key][0]) # Extract scalar

   # Plot the simple distribution using AnalysisG's internal Plotting module
   th = TH1F()
   th.xData = counts
   th.Title = "Number of High $p_T$ Tops per Event"
   th.xTitle = "Tops ($p_T > 500$ GeV)"
   th.yTitle = "Events"
   th.xBins = 10
   th.xMin = 0
   th.xMax = 10
   th.color = "royalblue"
   th.Filename = "high_pt_tops_distribution"
   th.SaveFigure()

   print(f"Saved plot to {th.Filename}.pdf!")

Congratulations! You have successfully traced the journey of an analysis from a raw ROOT `Particle` to a final physical distribution.
