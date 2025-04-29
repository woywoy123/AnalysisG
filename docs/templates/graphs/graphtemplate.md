The GraphTemplate Source Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to the `event_template` class, the `graph_template` class allows for graphs to be constructed, that can be later used to train a Graph Neural Network.
In this class, specific types of particles can be used to construct `Nodes`, which are subsequently assigned some set of input features.
Assignment of features is performed by passing lambda functions that specifically retrieve such features.
For instance, given some callable function :math:`y_i = \mathcal{F}(x_i)`, with input particle :math:`x_i` and output feature `y_i`, the template class will assign each particle node the value :math:`y_i` in a graph.
By a similar token, to construct edge features, the callable function can be functionally expressed as :math:`y_{ij} = \mathcal{F}(x_i, x_j)`.

To further simplify the pipeline, internal functions exploit dimensional analysis on the output vector space to infer the dimension of the `PyTorch` tensor and simply builds this tensor.

Example Implementation of a Custom Graph Class (`graph_name`)
=============================================================

This document details how to create a custom graph class, named `graph_name` in this example, by inheriting from the base `graph_template`. This allows for tailored graph construction logic within the analysis framework.

The implementation involves two files: a header (`.h`) and a source (`.cpp`).

Header File (`graph-name.h`)
----------------------------

The header file declares the structure of the custom graph class:

.. code-block:: C++

    #ifndef GRAPHS_GRAPHNAME_H
    #define GRAPHS_GRAPHNAME_H

    #include <event-name/some-event.h> // Include necessary event definition
    #include <graph_template.h>       // Include the base graph template

    class graph_name : public graph_template
    {
    public:
        // Default constructor
        graph_name();

        // Destructor
        ~graph_name();

        // Creates a copy of this graph object (polymorphism support)
        graph_template* clone() override;

        // Defines graph structure and features for a given event
        void CompileEvent() override;
    };

    #endif

Key elements defined:
  - **Inheritance:** `graph_name` publicly inherits from `graph_template`.
  - **Constructor/Destructor:** Standard `graph_name()` and `~graph_name()`.
  - **`clone()` Method:** An overridden virtual method. This is crucial for the framework to create instances of the specific `graph_name` type polymorphically when only a `graph_template` pointer is available.
  - **`CompileEvent()` Method:** An overridden virtual method. This is the core function where the user defines how to build the graph (nodes, edges, features) based on the input event data.

Source File (`graph-name.cpp`)
------------------------------

The source file provides the implementation for the methods declared in the header.

.. code-block:: C++

    #include "graph-name.h"

    // --- Boilerplate Code ---
    graph_name::graph_name() { this->name = "graph-name"; } // Set graph name
    graph_name::~graph_name() {}
    graph_template* graph_name::clone() { return (graph_template*)new graph_name(); } // Return new instance

    // --- Core Graph Building Logic ---
    void graph_name::CompileEvent() {
        // --- 1. Define Feature Extraction Lambdas ---
        // These functions extract specific values from event, particle, or particle-pair data.
        // They take a pointer to the output variable (`out`) and the relevant data object(s).

        // Graph-level truth feature (e.g., is this a signal event?)
        auto some_graph_fx = [](bool* out, event_name* ev) { *out = ev->truth_signal_variable; };

        // Node-level truth feature (e.g., particle type identification)
        auto some_node_fx = [](int* out, particle_template* p) { *out = p->is_top; };

        // Edge-level truth feature (e.g., do particles share a common origin?)
        auto some_edge_fx = [](int* out, std::tuple<particle_template*, particle_template*>* e_ij) {
            *out = std::get<0>(*e_ij)->top_index == std::get<1>(*e_ij)->top_index;
        };

        // Graph-level data feature (e.g., event-wide observable)
        auto some_other_graph_fx = [](double* out, event_name* ev) { *out = ev->missing_et; };

        // Node-level data feature (e.g., particle kinematics)
        auto some_other_node_fx = [](double* out, particle_template* p) { *out = p->pt; };

        // Edge-level data feature (e.g., relationship between connected particles)
        auto some_other_edge_fx = [](double* out, std::tuple<particle_template*, particle_template*>* e_ij) {
            *out = std::get<0>(*e_ij)->pt - std::get<1>(*e_ij)->pt;
        };

        // --- 2. Retrieve Event Data ---
        // Get the specific event object associated with this graph instance.
        event_name* event = this->get_event<event_name>();
        // Access the particles (nodes) within the event.
        std::vector<particle_template*> particles = event->some_var_with_particles; // Assuming 'some_var_with_particles' holds the node data

        // --- 3. Define Graph Topology (Optional) ---
        // Specify conditions for creating edges between nodes.
        // If omitted, a fully connected graph is assumed initially.
        // Example: Only connect particles if they are not both b-jets.
        auto bias_topology = [](particle_template* p1, particle_template* p2) {
            return !(p1->is_b && p2->is_b); // Example condition: don't connect two b-jets
        };
        // Apply the topology rule. Edges will only be considered between pairs (p1, p2) for which bias_topology(p1, p2) is true.
        this->define_topology(particles, bias_topology);

        // --- 4. Add Features to the Graph ---
        // Register the defined lambdas to populate graph features.
        // The base class handles storing these features and converting them later.
        // Each feature needs a unique name (string).

        // Truth Features (often used for training labels or auxiliary tasks)
        this->add_graph_truth_feature<bool, event_name>(event, some_graph_fx, "is_signal");
        this->add_node_truth_feature<int, particle_template>(particles, some_node_fx, "is_top");
        this->add_edge_truth_feature<int, particle_template>(some_edge_fx, "same_top"); // Applies only to edges allowed by topology

        // Data Features (observables used as input for GNNs)
        this->add_graph_data_feature<double, event_name>(event, some_other_graph_fx, "met");
        this->add_node_data_feature<double, particle_template>(particles, some_other_node_fx, "pt");
        this->add_edge_data_feature<double, particle_template>(some_other_edge_fx, "delta_pt"); // Applies only to edges allowed by topology

        // --- 5. Access Metadata (Optional) ---
        // Access information about the source data if needed.
        std::string root_file_path = this->filename; // Source filename
        event_t ev_data = this->data;               // Raw event data structure (if available/needed)
    }

**Explanation of `CompileEvent()`:**

This method orchestrates the graph construction for each event:

1.  **Define Feature Lambdas:** Small, anonymous functions capture the logic for extracting individual features. This promotes modularity. They accept a pointer `out` for the result and pointers to the relevant data structures (event, particle, or particle pair tuple).
2.  **Retrieve Event Data:** The `get_event<event_name>()` method fetches the correctly typed event object, providing access to all event information, including the list of particles that will become graph nodes.
3.  **Define Topology:** The `define_topology()` method (optional) filters potential edges. It takes a list of particles and a lambda function. The lambda should return `true` for pairs of particles (`p1`, `p2`) that *should* have an edge between them. If `define_topology` is not called, all possible pairs of nodes are initially considered connected (fully connected graph). Subsequent edge feature calculations (`add_edge_*_feature`) will only be performed for edges that satisfy this topology.
4.  **Add Features:** Methods like `add_graph_truth_feature`, `add_node_data_feature`, `add_edge_truth_feature`, etc., are used to associate the defined lambda functions with specific feature names.
    -   `graph_*`: Features associated with the entire graph.
    -   `node_*`: Features associated with each particle/node.
    -   `edge_*`: Features associated with connections between pairs of nodes (respecting the defined topology).
    -   `*_truth_*`: Typically used for labels or ground truth information.
    -   `*_data_*`: Typically used for input features/observables for a model.
    The template arguments specify the feature's data type (e.g., `bool`, `int`, `double`) and the type of object the lambda expects (e.g., `event_name`, `particle_template`). The base `graph_template` class manages the collection of these features for all nodes/edges.
5.  **Access Metadata:** Provides access to the original filename (`this->filename`) and potentially the raw event data structure (`this->data`) for more complex use cases.

By implementing `CompileEvent` in this structured way, users can define custom graph representations from their specific event data, leveraging the underlying `graph_template` framework for data handling and eventual conversion to formats suitable for GNN libraries like PyTorch.

The above code is all that is required to construct graphs for Graph Neural Network training!
There are no additional steps required to start using the `PyTorch` API.

For more information about the methods and attributes of the `graph_template` class, see the core-class documentation :ref:`graph-template`.
