# Graph Representation of Physics Events

Graph-based representations are at the core of AnalysisG's approach to physics event analysis. This section explains how physics events are transformed into graph structures suitable for machine learning.

## Conceptual Overview

In high energy physics, events are collections of particles produced in collisions. These events can be naturally represented as graphs, where:

- **Nodes**: Represent individual particles or detector objects
- **Edges**: Represent relationships or interactions between particles
- **Features**: Physical properties associated with nodes and edges

Graph representations capture both the properties of individual particles and their relationships, providing a rich structure for machine learning algorithms to analyze.

## Graph Structure in AnalysisG

The `graph_template` class provides the foundation for graph representation in AnalysisG:

```cpp
graph_template* graph = new graph_template();
```

A graph consists of the following components:

- **Nodes**: Particles or physics objects in the event
- **Edges**: Connections between nodes, defined by topology
- **Node features**: Properties of each node (e.g., pt, eta, phi)
- **Edge features**: Properties of connections between nodes (e.g., deltaR)
- **Graph features**: Global properties of the entire graph (e.g., total energy)

## Building a Graph from Event Data

To create a graph from an event:

```cpp
void build_graph(event_template* event) {
    graph_template* graph = new graph_template();
    
    // Build the graph from the event
    // This associates particles in the event with graph nodes
    graph->build(event);
    
    // Continue with feature definition and topology...
}
```

## Feature Types

AnalysisG distinguishes between two types of features:

1. **Data Features**: Measurable properties from detector data
   ```cpp
   // Add node-level data features
   graph->add_node_data_feature<double>(particle_template::pt, "pt");
   graph->add_node_data_feature<double>(particle_template::eta, "eta");
   graph->add_node_data_feature<double>(particle_template::phi, "phi");
   
   // Add edge-level data features
   graph->add_edge_data_feature<double>(calculate_deltaR, "deltaR");
   ```

2. **Truth Features**: Properties from simulation (for training)
   ```cpp
   // Add truth features
   graph->add_node_truth_feature<int>(particle_template::is_signal, "is_signal");
   graph->add_graph_truth_feature<int>(event_template::event_class, "event_class");
   ```

## Graph Topologies

AnalysisG supports several graph topology types:

### Fully Connected Graph

Every node is connected to every other node:

```cpp
graph->define_topology(topology_type::fully_connected);
```

### k-Nearest Neighbors

Each node is connected to its k nearest neighbors in η-φ space:

```cpp
graph->define_topology(topology_type::knn, 5);  // 5-nearest neighbors
```

### Custom Topology

Define custom connection rules:

```cpp
graph->define_topology([](particle_template* p1, particle_template* p2) -> bool {
    // Connect only particles within deltaR < 0.4
    double deltaR = p1->deltaR(*p2);
    return deltaR < 0.4;
});
```

## Example: Complete Graph Definition

Here's a complete example of defining a graph for a top tagging analysis:

```cpp
graph_template* create_top_tagging_graph(event_template* event) {
    graph_template* graph = new graph_template();
    
    // Build graph from event
    graph->build(event);
    
    // Add node features (properties of individual particles)
    graph->add_node_data_feature<double>(particle_template::pt, "pt");
    graph->add_node_data_feature<double>(particle_template::eta, "eta");
    graph->add_node_data_feature<double>(particle_template::phi, "phi");
    graph->add_node_data_feature<double>(particle_template::energy, "energy");
    graph->add_node_data_feature<double>(particle_template::btag, "btag");
    
    // Add edge features (properties of particle relationships)
    graph->add_edge_data_feature<double>(calculate_deltaR, "deltaR");
    graph->add_edge_data_feature<double>(calculate_deltaEta, "deltaEta");
    graph->add_edge_data_feature<double>(calculate_deltaPhi, "deltaPhi");
    
    // Add graph features (global properties of the event)
    graph->add_graph_data_feature<int>(event_template::njets, "njets");
    graph->add_graph_data_feature<double>(event_template::met, "met");
    
    // Add truth features (for training)
    graph->add_node_truth_feature<int>(particle_template::parton_id, "parton_id");
    graph->add_graph_truth_feature<int>(event_template::is_top, "is_top");
    
    // Define topology (k-nearest neighbors)
    graph->define_topology(topology_type::knn, 8);
    
    return graph;
}
```

## Data Export for Machine Learning

Once a graph is constructed, it can be exported to tensor formats for machine learning:

```cpp
// Export graph to torch tensors
torch::Tensor node_features = graph->export_node_features();
torch::Tensor edge_features = graph->export_edge_features();
torch::Tensor graph_features = graph->export_graph_features();
torch::Tensor adjacency = graph->export_adjacency();

// Or use the built-in exporter
graph_data_t graph_data = graph->data_export();
```

## Graph Registration in Analysis

To use a graph template in your analysis:

```cpp
analysis* an = new analysis();

// Register graph template with the analysis
an->set_graph_template([](event_template* event) -> graph_template* {
    return create_top_tagging_graph(event);
});
```

## Best Practices

1. **Feature Selection**: Include only relevant physical features to avoid overfitting
2. **Normalization**: Consider normalizing features that span different scales
3. **Topology**: Choose a topology appropriate for your physics problem
4. **Memory Management**: Be aware of memory usage for large graphs
5. **Feature Engineering**: Create derived features that encode physics knowledge