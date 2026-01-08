graph Module
============

The ``graph`` module provides utilities for constructing and manipulating physics graphs for machine learning applications.

Graph Construction
------------------

.. py:function:: build_graph(nodes, edge_indices=None, k_nearest=None)

   Construct a graph from node features and optional edge information.
   
   :param torch.Tensor nodes: Node feature tensor with shape [num_nodes, num_features]
   :param torch.Tensor edge_indices: Optional edge indices with shape [2, num_edges]
   :param int k_nearest: Optional number of nearest neighbors for kNN graph construction
   :return: Dictionary containing graph information
   :rtype: dict


Feature Extraction
------------------

.. py:function:: node_features(four_momenta, particle_ids=None)

   Extract physics-inspired node features from four-momenta.
   
   :param torch.Tensor four_momenta: Four-momenta tensor with shape [num_nodes, 4]
   :param torch.Tensor particle_ids: Optional particle ID tensor with shape [num_nodes]
   :return: Node feature tensor
   :rtype: torch.Tensor

.. py:function:: edge_features(four_momenta, edge_indices)

   Compute physics-inspired edge features from node four-momenta and edge indices.
   
   :param torch.Tensor four_momenta: Four-momenta tensor with shape [num_nodes, 4]
   :param torch.Tensor edge_indices: Edge indices tensor with shape [2, num_edges]
   :return: Edge feature tensor
   :rtype: torch.Tensor

.. py:function:: global_features(four_momenta)

   Compute global features from all node four-momenta.
   
   :param torch.Tensor four_momenta: Four-momenta tensor with shape [num_nodes, 4]
   :return: Global feature tensor
   :rtype: torch.Tensor

Graph Transformations
---------------------

.. py:function:: to_undirected(edge_indices, edge_attr=None)

   Convert a directed graph to an undirected graph by adding reverse edges.
   
   :param torch.Tensor edge_indices: Edge indices tensor with shape [2, num_edges]
   :param torch.Tensor edge_attr: Optional edge attributes with shape [num_edges, num_features]
   :return: Undirected edge indices and attributes
   :rtype: tuple(torch.Tensor, torch.Tensor)

.. py:function:: subgraph(node_subset, edge_indices, edge_attr=None, relabel_nodes=True)

   Extract a subgraph containing only the specified nodes.
   
   :param torch.Tensor node_subset: Indices of nodes to keep
   :param torch.Tensor edge_indices: Edge indices tensor with shape [2, num_edges]
   :param torch.Tensor edge_attr: Optional edge attributes with shape [num_edges, num_features]
   :param bool relabel_nodes: Whether to relabel nodes to have consecutive indices
   :return: Subgraph edge indices and attributes
   :rtype: tuple(torch.Tensor, torch.Tensor)

Examples
--------

Basic usage examples:

.. code-block:: python

   import torch
   from AnalysisG.pyc import graph
   
   # Create some example four-momenta (E, px, py, pz)
   four_momenta = torch.tensor([
       [100.0, 10.0, 20.0, 30.0],
       [120.0, -15.0, 25.0, 35.0],
       [80.0, 5.0, -10.0, 20.0],
       [150.0, 30.0, 40.0, 50.0]
   ])
   
   # Extract node features
   node_feats = graph.node_features(four_momenta)
   print(f"Node features shape: {node_feats.shape}")
   
   # Create a kNN graph using build_graph
   knn_graph_data = graph.build_graph(nodes=node_feats, k_nearest=2)
   edge_indices = knn_graph_data['edge_index']
   print(f"Edge indices: {edge_indices}")
   
   # Extract edge features
   edge_feats = graph.edge_features(four_momenta, edge_indices)
   print(f"Edge features shape: {edge_feats.shape}")
   
   # Create a complete graph with physics-inspired features
   physics_graph = graph.build_graph(four_momenta)
   print(f"Graph nodes: {physics_graph['num_nodes']}")
   print(f"Graph edges: {physics_graph['num_edges']}")
   
   # Convert directed graph to undirected
   undirected_edges, undirected_attrs = graph.to_undirected(
       edge_indices, edge_feats
   )
   print(f"Undirected edges: {undirected_edges.shape[1]}")
   
   # Extract subgraph
   node_subset = torch.tensor([0, 2])
   subgraph_edges, subgraph_attrs = graph.subgraph(
       node_subset, edge_indices, edge_feats
   )
   print(f"Subgraph edges: {subgraph_edges.shape[1]}")