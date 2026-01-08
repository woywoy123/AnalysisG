.. _tutorials_ttbar:

=============================
Top Quark Pair Classification
=============================

This tutorial shows how to use AnalysisG for a typical HEP analysis: classifying top quark pair (tt̄) events.

Objective
----------

We will train a Graph Neural Network (GNN) to distinguish tt̄ events from background signals. The steps are:

1. Define an event template for tt̄ events.
2. Create a graph template for the graph representation.
3. Train a GNN classifier.
4. Evaluate the results.

Prerequisites
-------------

- AnalysisG installed (see :ref:`installation`).
- ROOT files with tt̄ events and background processes.
- Basic knowledge of particle physics and tt̄ decay.

Event Template
---------------

First, we define an event template to extract relevant objects from the ROOT files:

.. code-block:: cpp

    class TTbarEvent : public event_template
    {
    public:
         TTbarEvent()
         {
              this->name = "TTbarEvent";
              jets = new std::vector<Jet>();
              electrons = new std::vector<Electron>();
              muons = new std::vector<Muon>();
         }

         ~TTbarEvent()
         {
              delete jets;
              delete electrons;
              delete muons;
         }

         void ReadFrom(TTree* tree) override
         {
              // Branches for jets
              tree->SetBranchAddress("jet_pt", &branch_jet_pt);
              tree->SetBranchAddress("jet_eta", &branch_jet_eta);
              tree->SetBranchAddress("jet_phi", &branch_jet_phi);
              tree->SetBranchAddress("jet_e", &branch_jet_e);
              tree->SetBranchAddress("jet_btag", &branch_jet_btag);

              // Branches for leptons
              tree->SetBranchAddress("el_pt", &branch_el_pt);
              tree->SetBranchAddress("el_eta", &branch_el_eta);
              // ... other branches ...

              // Event-level variables
              tree->SetBranchAddress("met", &met);
              tree->SetBranchAddress("met_phi", &met_phi);

              // MC truth (if available)
              if (tree->GetBranch("mc_is_ttbar"))
                    tree->SetBranchAddress("mc_is_ttbar", &is_ttbar);
         }

         void ProcessEntry() override
         {
              // Process jets
              jets->clear();
              for (size_t i = 0; i < branch_jet_pt->size(); i++)
              {
                    Jet j;
                    j.pt = branch_jet_pt->at(i);
                    j.eta = branch_jet_eta->at(i);
                    j.phi = branch_jet_phi->at(i);
                    j.e = branch_jet_e->at(i);
                    j.btag = branch_jet_btag->at(i);
                    jets->push_back(j);
              }

              // Process electrons and muons similarly
              // ...
         }

         // Data structures
         std::vector<Jet>* jets;
         std::vector<Electron>* electrons;
         std::vector<Muon>* muons;
         float met;
         float met_phi;
         bool is_ttbar;

    private:
         // Branch pointers
         std::vector<float>* branch_jet_pt;
         std::vector<float>* branch_jet_eta;
         // ... other branch pointers ...
    };

Graph Template
-------------

Next, we define a graph template to create the graph structure for our GNN:

.. code-block:: cpp

    class TTbarGraph : public graph_template
    {
    public:
         TTbarGraph()
         {
              this->name = "ttbar_graph";
         }

         ~TTbarGraph() {}

         graph_template* clone() override
         {
              return (graph_template*)new TTbarGraph();
         }

         void CompileEvent() override
         {
              // Get the event
              TTbarEvent* ev = this->get_event<TTbarEvent>();

              // Create nodes for all physics objects
              std::vector<particle_template*> particles;

              // Add jets as nodes
              for (size_t i = 0; i < ev->jets->size(); i++)
              {
                    particles.push_back(&ev->jets->at(i));
              }

              // Add leptons as nodes
              for (size_t i = 0; i < ev->electrons->size(); i++)
              {
                    particles.push_back(&ev->electrons->at(i));
              }

              for (size_t i = 0; i < ev->muons->size(); i++)
              {
                    particles.push_back(&ev->muons->at(i));
              }

              this->define_particle_nodes(&particles);

              // Graph feature: Missing ET
              auto get_met = [](float* out_val, TTbarEvent* ev) {*out_val = ev->met;};
              this->add_graph_data_feature<float, TTbarEvent>(ev, get_met, "met");

              auto get_met_phi = [](float* out_val, TTbarEvent* ev) {*out_val = ev->met_phi;};
              this->add_graph_data_feature<float, TTbarEvent>(ev, get_met_phi, "met_phi");

              // Node features for jets
              auto get_pt = [](float* out_val, Jet* jet) {*out_val = jet->pt;};
              this->add_node_data_feature<float, Jet>(get_pt, "pt");

              auto get_eta = [](float* out_val, Jet* jet) {*out_val = jet->eta;};
              this->add_node_data_feature<float, Jet>(get_eta, "eta");

              auto get_phi = [](float* out_val, Jet* jet) {*out_val = jet->phi;};
              this->add_node_data_feature<float, Jet>(get_phi, "phi");

              auto get_btag = [](float* out_val, Jet* jet) {*out_val = jet->btag;};
              this->add_node_data_feature<float, Jet>(get_btag, "btag");

              // Similar features for electrons and muons
              // ...

              // Define topology: Fully connected graph
              auto edge_rule = [](particle_template* p1, particle_template* p2) {
                    return true;  // All particles are connected
              };
              this->define_topology(edge_rule);

              // Edge features: Delta R between particles
              auto get_delta_r = [](float* out_val, std::tuple<particle_template*, particle_template*>* pair) {
                    particle_template* p1 = std::get<0>(*pair);
                    particle_template* p2 = std::get<1>(*pair);

                    float delta_eta = p1->eta - p2->eta;
                    float delta_phi = std::abs(p1->phi - p2->phi);
                    if (delta_phi > M_PI) delta_phi = 2*M_PI - delta_phi;

                    *out_val = std::sqrt(delta_eta*delta_eta + delta_phi*delta_phi);
              };
              this->add_edge_data_feature<float, particle_template>(get_delta_r, "delta_r");

              // Add label (if MC truth is available)
              auto get_label = [](int* out_val, TTbarEvent* ev) {*out_val = ev->is_ttbar ? 1 : 0;};
              this->add_graph_class_feature<int, TTbarEvent>(ev, get_label, "is_ttbar");
         }

         bool PreSelection() override
         {
              // Select events with at least 4 jets and at least 1 lepton
              TTbarEvent* ev = this->get_event<TTbarEvent>();
              return (ev->jets->size() >= 4 && (ev->electrons->size() + ev->muons->size() >= 1));
         }
    };

Python Script for Analysis
---------------------------

Now we can write a Python script using our templates to run the analysis:

.. code-block:: python

    from AnalysisG.core import analysis
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool

    # Define GNN model
    class TTbarGNN(torch.nn.Module):
         def __init__(self, node_features):
              super(TTbarGNN, self).__init__()
              self.conv1 = GCNConv(node_features, 64)
              self.conv2 = GCNConv(64, 64)
              self.conv3 = GCNConv(64, 64)
              self.fc1 = torch.nn.Linear(64, 32)
              self.fc2 = torch.nn.Linear(32, 2)  # Binary classification

         def forward(self, data):
              x, edge_index, batch = data.x, data.edge_index, data.batch

              # Graph Convolutions
              
