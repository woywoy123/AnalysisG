.. _graph_templates:

=================
Graph Templates
=================

The `graph_template` class is fundamental for creating graph representations of physics events for Graph Neural Networks (GNNs) within the AnalysisG framework.

Core Concept
------------

A graph object encapsulates a single physics event, structured as:

*   **Nodes**: Represent particles or other physics objects (e.g., jets, leptons).
*   **Edges**: Define relationships or connections between nodes.
*   **Features**: Store properties associated with nodes, edges, or the entire graph (event-level information).

Implementing a Graph Template
-----------------------------

To define how events are converted into graphs, you inherit from `graph_template`. Here’s a conceptual example:

.. code-block:: cpp

     #include "AnalysisG/Templates/graph_template.h"
     #include "YourEventFormat.h" // Include your event data structure header

     class MyCustomGraph : public graph_template
     {
     public:
           // Constructor: Set a unique name for this graph type
           MyCustomGraph()
           {
                 this->name = "my_custom_graph";
           }

           // Destructor
           ~MyCustomGraph() {}

           // Required: Method to create a new instance of this template
           graph_template* clone() override
           {
                 return (graph_template*)new MyCustomGraph();
           }

           // Required: Main method to build the graph from event data
           void CompileEvent() override
           {
                 // --- 1. Get Event Data ---
                 // Access the event data associated with this graph instance
                 MyEventFormat* event = this->get_event<MyEventFormat>();

                 // --- 2. Define Nodes ---
                 // Select particles/objects to become nodes
                 std::vector<particle_template*> node_particles;
                 for (size_t i = 0; i < event->jets.size(); ++i) // Example: Use jets as nodes
                 {
                         node_particles.push_back(&event->jets[i]);
                 }
                 // Create nodes in the graph structure
                 this->define_particle_nodes(&node_particles);

                 // --- 3. Add Graph-Level Features ---
                 // Features describing the whole event (e.g., Missing Transverse Energy)
                 auto get_met = [](float* out_val, MyEventFormat* ev) { *out_val = ev->met; };
                 this->add_graph_data_feature<float, MyEventFormat>(event, get_met, "met");

                 // --- 4. Add Node-Level Features ---
                 // Features describing individual nodes (e.g., Jet pT)
                 auto get_jet_pt = [](float* out_val, Jet* jet) { *out_val = jet->pt; };
                 this->add_node_data_feature<float, Jet>(get_jet_pt, "pt");
                 // Add other node features (eta, phi, energy, particle type flags, etc.)

                 // --- 5. Define Topology (Edges) ---
                 // Specify how nodes are connected
                 // Example: Fully connected graph (all nodes connect to all others)
                 auto connect_all = [](particle_template* p1, particle_template* p2) {
                         return true; // Connect every pair
                 };
                 this->define_topology(connect_all);
                 // Alternative: Connect based on proximity (e.g., Delta R < threshold)
                 // auto connect_nearby = ... (define rule based on particle properties)
                 // this->define_topology(connect_nearby);

                 // --- 6. Add Edge-Level Features ---
                 // Features describing the connections (e.g., Delta R between connected jets)
                 auto get_delta_r = [](float* out_val, std::tuple<Jet*, Jet*>* jet_pair) {
                         Jet* j1 = std::get<0>(*jet_pair);
                         Jet* j2 = std::get<1>(*jet_pair);
                         // Calculate Delta R between j1 and j2...
                         float dEta = j1->eta - j2->eta;
                         float dPhi = std::abs(j1->phi - j2->phi);
                         if (dPhi > M_PI) dPhi = 2 * M_PI - dPhi; // Handle phi wrap-around
                         *out_val = std::sqrt(dEta * dEta + dPhi * dPhi);
                 };
                 this->add_edge_data_feature<float, Jet>(get_delta_r, "delta_r");
                 // Add other edge features (invariant mass of pair, angular separation, etc.)
           }

           // Optional: Method to filter events before graph construction
           bool PreSelection() override
           {
                 MyEventFormat* event = this->get_event<MyEventFormat>();
                 // Example: Require at least 4 jets
                 return event->jets.size() >= 4;
           }
     };

Key Implementation Steps
------------------------

1.  **Define Nodes**: Select which physics objects from the event data will become nodes in the graph.
     .. code-block:: cpp

          std::vector<particle_template*> particles;
          // Populate 'particles' with pointers to your chosen objects (jets, leptons, etc.)
          this->define_particle_nodes(&particles);

2.  **Add Features**: Attach numerical data to the graph, nodes, or edges.
     *   **Graph Features**: Event-wide properties.
          .. code-block:: cpp
               auto get_feature = [](float* out, EventClass* ev) { /* extract feature */; };
               this->add_graph_data_feature<float, EventClass>(ev_ptr, get_feature, "feature_name");

     *   **Node Features**: Properties of individual particles/nodes.
          .. code-block:: cpp
               auto get_feature = [](float* out, ParticleClass* p) { /* extract feature */; };
               this->add_node_data_feature<float, ParticleClass>(get_feature, "feature_name");

     *   **Edge Features**: Properties calculated from pairs of connected nodes.
          .. code-block:: cpp
               auto get_feature = [](float* out, std::tuple<ParticleClass*, ParticleClass*>* pair) { /* calculate feature */; };
               this->add_edge_data_feature<float, ParticleClass>(get_feature, "feature_name");

3.  **Define Topology**: Specify the criteria for creating edges between nodes.
     .. code-block:: cpp
          // Define a rule (lambda function) that returns true if two particles should be connected
          auto connection_rule = [](particle_template* p1, particle_template* p2) {
                // Example: Connect if Delta R < 1.0
                // Cast p1 and p2 to their actual types (e.g., Jet*) to access properties
                // Calculate distance or apply other logic...
                // return distance < 1.0;
                return true; // Simplest case: fully connected
          };
          this->define_topology(connection_rule);

4.  **(Optional) PreSelection**: Implement the `PreSelection` method to efficiently skip events that don't meet certain criteria before attempting graph construction.

Registering and Using the Template
----------------------------------

Your custom graph template needs to be associated with the corresponding event data format within your AnalysisG Python script.

.. code-block:: python

     # In your Python analysis script:
     from AnalysisG import Analysis
     # Import your custom C++ event and graph classes (via PyROOT bindings)
     # from AnalysisG.events import MyEvent # Assuming bindings are set up
     # from AnalysisG.graphs import MyCustomGraph # Assuming bindings are set up

     # Instantiate the event handler and graph template
     # event_handler = MyEvent()
     # graph_builder = MyCustomGraph()
     # graph_builder.PreSelection = True # Enable pre-selection if defined

     # Create the main Analysis object
     ana = Analysis()
     # Configure paths, threads, etc.
     # ana.InputDirectory = "/path/to/your/data"
     # ana.OutputDirectory = "./analysis_output"
     # ana.Threads = 8

     # Add the event handler and graph template for a specific dataset tag
     # ana.AddEvent(event_handler, "dataset_tag")
     # ana.AddGraph(graph_builder, "dataset_tag")

     # Add samples associated with the tag
     # ana.AddSamples(ana.InputDirectory + "/*.root", "dataset_tag")

     # Start the analysis (graph generation)
     # ana.Start()

Using Graphs for Machine Learning
---------------------------------

Once graphs are generated, they can be fed into GNN models for training or inference.

.. code-block:: python

     # (Continuing the Python script)
     from AnalysisG.models import SomeGNNModel # Import a GNN model architecture
     from AnalysisG.core.lossfx import OptimizerConfig

     # --- Model Configuration ---
     # model = SomeGNNModel()
     # Configure model inputs (matching feature names from your graph template)
     # model.i_node = ["pt", "eta", "phi", ...]
     # model.i_edge = ["delta_r", ...]
     # model.i_graph = ["met", ...]
     # Configure model outputs and loss functions
     # model.o_graph = {"graph_classification_task": "CrossEntropyLoss"}
     # model.device = "cuda" # or "cpu"

     # --- Optimizer Configuration ---
     # optimizer_cfg = OptimizerConfig(Optimizer = "adam", lr = 1e-4)

     # --- Analysis Setup (ML part) ---
     # ana.AddGNNModel(model, optimizer_cfg, "my_gnn_model")
     # ana.TrainingDataset = "./ml_datasets/training" # Where to store datasets
     # ana.GraphCache = "./graph_cache" # Where graphs are stored/loaded from
     # ana.Epochs = 50
     # ana.kFolds = 5 # Optional: Use k-fold cross-validation
     # ana.Targets = ["graph_classification_task"] # Primary target for training/optimization

     # --- Run Training ---
     # ana.Start() # This will now include GNN training if configured

Further Resources
-----------------

*   API Documentation: :ref:`API-Graph-Template <api_reference/graph>`
*   Examples: Check the `/docs/examples` directory in the AnalysisG repository.

.. toctree::
   :maxdepth: 1
   :hidden:

   graphtemplate
   graph_example1
   graph_example2
