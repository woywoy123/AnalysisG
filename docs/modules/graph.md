# Graph Module

@brief Graph-based data representation and manipulation framework

## Overview

The Graph module provides a flexible system for representing and manipulating graph-based data structures. It is designed to handle complex relationships between entities, such as particles in physics events.

## Key Components

### graph_template

Base class for graph representations, providing core functionality for managing nodes, edges, and features.

```cpp
class graph_template: public tools
{
    // ...existing code...
    void add_graph_data_feature(O* ev, X fx, std::string name);
    void add_graph_truth_feature(O* ev, X fx, std::string name);
    // ...existing code...
};
```

### graph_t

Struct for managing graph data, including event weights and batched events.

```cpp
struct graph_t {
    torch::Tensor* get_event_weight(g* mdl);
    template <typename g>
    torch::Tensor* get_batch_index(g* mdl);
    template <typename g>
    torch::Tensor* get_batched_events(g* mdl);
    // ...existing code...
};
```

## Usage Example

```cpp
// Create a graph object
graph_template* graph = new graph_template();

// Add data features
graph->add_graph_data_feature(event, missing_et, "met");
graph->add_graph_data_feature(event, num_jets, "num_jets");

// Add truth features
graph->add_graph_truth_feature(event, event_weight, "weight");

// Process the graph
graph->CompileEvent();
```

## Advanced Features

- **Feature Management**: Add, remove, and query features for nodes, edges, and graphs.
- **Batch Processing**: Efficiently handle batched graph data for machine learning models.
- **Integration with PyTorch**: Seamless integration with PyTorch tensors for deep learning applications.