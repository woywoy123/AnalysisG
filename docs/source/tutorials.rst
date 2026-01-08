=========
Tutorials
=========

This section contains detailed tutorials for various aspects of AnalysisG.

Available Tutorials
===================

.. contents::
   :local:
   :depth: 2

Event Templates
===============

Creating Custom Event Templates
-------------------------------

Event templates define how data is read from input files. Here's a comprehensive guide:

.. code-block:: cpp

    class TTbarEvent : public event_template {
    public:
        // Particle collections
        std::vector<Electron*> electrons;
        std::vector<Muon*> muons;
        std::vector<Jet*> jets;
        
        // Event-level variables
        double met;
        double met_phi;
        int n_jets;
        
        TTbarEvent() {
            // Set template name
            name = "ttbar_event";
            
            // Define tree to read
            add_tree("nominal");
            
            // Define branches for particles
            add_branch("electrons", "el_");
            add_branch("muons", "mu_");
            add_branch("jets", "jet_");
            
            // Define leaves for event variables
            add_leaf("met", "met_et");
            add_leaf("met_phi", "met_phi");
            add_leaf("n_jets", "jet_n");
            
            // Register particle generators
            register_particle("electrons", Electron::create);
            register_particle("muons", Muon::create);
            register_particle("jets", Jet::create);
        }
        
        void build(element_t* el) override {
            // Extract event-level data
            met = el->data["met"]->get<double>();
            met_phi = el->data["met_phi"]->get<double>();
            n_jets = el->data["n_jets"]->get<int>();
            
            // Particles are automatically built from registered generators
        }
        
        TTbarEvent* clone() override {
            return new TTbarEvent();
        }
    };

Graph Templates
===============

Building Graph Representations
------------------------------

Graph templates transform events into graph structures suitable for ML:

.. code-block:: cpp

    class JetGraph : public graph_template {
    public:
        JetGraph() {
            name = "jet_graph";
            
            // Define edge topology: fully connected
            define_topology([](particle_template* p1, particle_template* p2) {
                return true;
            });
        }
        
        void CompileEvent() override {
            TTbarEvent* ev = get_event<TTbarEvent>();
            
            // Define which particles become nodes
            define_particle_nodes(&ev->jets);
            
            // Add node features (per jet)
            add_node_data_feature<double, Jet>(
                [](Jet* j) { return j->pt() / 1000.0; },  // Scale to GeV
                "pt"
            );
            
            add_node_data_feature<double, Jet>(
                [](Jet* j) { return j->eta(); },
                "eta"
            );
            
            add_node_data_feature<double, Jet>(
                [](Jet* j) { return j->phi(); },
                "phi"
            );
            
            // Add edge features (per jet pair)
            add_edge_data_feature<double, Jet>(
                [](std::tuple<Jet*, Jet*>* pair) {
                    Jet* j1 = std::get<0>(*pair);
                    Jet* j2 = std::get<1>(*pair);
                    return deltaR(j1, j2);
                },
                "deltaR"
            );
            
            // Add graph-level features
            add_graph_data_feature<double, TTbarEvent>(
                ev,
                [](TTbarEvent* e) { return e->met / 1000.0; },
                "met"
            );
            
            // Add truth labels for training
            add_node_truth_feature<int, Jet>(
                [](Jet* j) { return j->isFromTop ? 1 : 0; },
                "from_top"
            );
        }
        
        bool PreSelection() override {
            // Require at least 4 jets for ttbar
            return num_nodes >= 4;
        }
    };

Model Templates
===============

Creating Neural Network Models
------------------------------

Model templates define your neural network architecture:

.. code-block:: cpp

    class JetClassifier : public model_template {
    public:
        torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
        
        JetClassifier() {
            // Define model name
            name = "jet_classifier";
            
            // Register network layers
            fc1 = register_module("fc1", torch::nn::Linear(3, 64));
            fc2 = register_module("fc2", torch::nn::Linear(64, 32));
            fc3 = register_module("fc3", torch::nn::Linear(32, 1));
        }
        
        torch::Tensor forward(graph_t* data) override {
            // Get node features: [N_nodes, 3]
            auto pt = *data->get_data_node("pt", this);
            auto eta = *data->get_data_node("eta", this);
            auto phi = *data->get_data_node("phi", this);
            
            // Stack features
            auto x = torch::stack({pt, eta, phi}, 1);
            
            // Forward pass
            x = torch::relu(fc1->forward(x));
            x = torch::relu(fc2->forward(x));
            x = torch::sigmoid(fc3->forward(x));
            
            return x;
        }
    };

Training Configuration
----------------------

Configure training with optimizer parameters:

.. code-block:: cpp

    optimizer_params_t config;
    config.learning_rate = 0.001;
    config.epochs = 100;
    config.batch_size = 64;
    config.optimizer = "Adam";
    config.lr_scheduler = "StepLR";
    config.lr_step_size = 30;
    config.lr_gamma = 0.1;
    config.early_stopping = true;
    config.patience = 10;
    config.k_folds = 5;

Selections
==========

Implementing Event Selections
-----------------------------

Selection templates filter events based on physics criteria:

.. code-block:: cpp

    class DileptonSelection : public selection_template {
    public:
        DileptonSelection() {
            name = "dilepton";
        }
        
        bool selection(event_template* event) override {
            TTbarEvent* ev = (TTbarEvent*)event;
            
            // Count leptons
            int n_el = 0, n_mu = 0;
            
            for (auto* el : ev->electrons) {
                if (el->pt() > 25000 && fabs(el->eta()) < 2.5) {
                    n_el++;
                }
            }
            
            for (auto* mu : ev->muons) {
                if (mu->pt() > 25000 && fabs(mu->eta()) < 2.5) {
                    n_mu++;
                }
            }
            
            // Require exactly 2 leptons (ee, mumu, or emu)
            return (n_el + n_mu) == 2;
        }
    };

    class JetSelection : public selection_template {
    public:
        JetSelection() {
            name = "jets";
        }
        
        bool selection(event_template* event) override {
            TTbarEvent* ev = (TTbarEvent*)event;
            
            int n_jets = 0;
            for (auto* j : ev->jets) {
                if (j->pt() > 25000 && fabs(j->eta()) < 2.5) {
                    n_jets++;
                }
            }
            
            return n_jets >= 2;
        }
    };

Complete Analysis
=================

Putting It All Together
-----------------------

Here's a complete analysis setup:

.. code-block:: cpp

    #include <AnalysisG/analysis.h>

    int main() {
        // Create analysis instance
        analysis ana;
        
        // Configure settings
        ana.m_settings.output = "results/";
        ana.m_settings.threads = 8;
        ana.m_settings.batch_size = 64;
        
        // Add data samples
        ana.add_samples("/data/ttbar_mc/", "ttbar");
        ana.add_samples("/data/background/", "bkg");
        
        // Add event templates
        ana.add_event_template(new TTbarEvent(), "ttbar");
        ana.add_event_template(new TTbarEvent(), "bkg");
        
        // Add selections
        ana.add_selection_template(new DileptonSelection());
        ana.add_selection_template(new JetSelection());
        
        // Add graph templates
        ana.add_graph_template(new JetGraph(), "ttbar");
        ana.add_graph_template(new JetGraph(), "bkg");
        
        // Configure training
        optimizer_params_t config;
        config.learning_rate = 0.001;
        config.epochs = 50;
        config.batch_size = 64;
        config.k_folds = 5;
        
        // Add model
        ana.add_model(new JetClassifier(), &config, "training");
        
        // Start analysis
        ana.start();
        
        // Monitor progress
        while (!ana.is_complete()["training"]) {
            auto progress = ana.progress();
            std::cout << "Epoch: " << progress["training"][0] << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
        
        return 0;
    }
