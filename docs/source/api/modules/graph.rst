Graph Module
===========

The Graph module is a fundamental component of the AnalysisG framework, responsible for representing physics events as graphs with nodes, edges, and features.

Overview
--------

This module implements the graph data structure used throughout the framework, providing functionalities for defining graph topology, adding node and edge features, and exporting graph data for machine learning pipelines.

Key Components
-------------

graph_template class
~~~~~~~~~~~~~~~~~~

.. doxygenclass:: graph_template
   :members:
   :protected-members:
   :undoc-members:

Main Functionalities
-------------------

Graph Construction
~~~~~~~~~~~~~~~~

The module provides methods to construct graph representations of physics events:

- Node definition and association with physical particles
- Edge construction between nodes based on physical criteria
- Graph-level feature extraction from event properties

Feature Management
~~~~~~~~~~~~~~~~

The module supports adding different types of features to the graph:

- ``add_graph_feature()``: Adds features at the graph level
- ``add_node_feature()``: Adds features to individual nodes
- ``add_edge_feature()``: Adds features to edges between nodes

Feature Types
~~~~~~~~~~~

Features are categorized into two main types:

- **Truth features**: Properties from simulation or ground truth (prefixed with "T-")
- **Data features**: Observable properties from detector measurements (prefixed with "D-")

Topology Definition
~~~~~~~~~~~~~~~~~

The module provides methods to define the graph topology:

- ``define_topology()``: Creates the adjacency structure of the graph
- Support for various topological patterns (fully connected, nearest neighbors, etc.)

Data Export
~~~~~~~~~

Graph data can be exported to formats suitable for machine learning:

- ``data_export()``: Converts the internal graph representation to tensor-based format
- Integration with PyTorch data structures for seamless model training

Usage Example
------------

.. code-block:: cpp

    #include <templates/graph_template.h>
    #include <templates/event_template.h>

    // Define a graph for a physics event
    graph_template* create_graph(event_template* event) {
        // Create a new graph
        graph_template* graph = new graph_template();
        
        // Build the graph from the event
        graph->build(event);
        
        // Add node features
        graph->add_node_data_feature<double, particle_template, decltype(pt)>(pt, "pt");
        graph->add_node_data_feature<double, particle_template, decltype(eta)>(eta, "eta");
        graph->add_node_data_feature<double, particle_template, decltype(phi)>(phi, "phi");
        
        // Add truth features for training
        graph->add_node_truth_feature<int, particle_template, decltype(is_signal)>(is_signal, "signal");
        
        // Define the graph topology (fully connected in this example)
        graph->define_topology(fulltopo);
        
        return graph;
    }