Graph Module (PyC)
==================

Graph algorithms and operations for graph neural networks.

Overview
--------

Located in ``pyc/pyc/graph/``, this module implements graph algorithms with 
optimized C++ and CUDA backends.

Key Functions
-------------

Edge Construction
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pyc.graph as graph
   
   # K-nearest neighbors graph
   edge_index = graph.knn_graph(node_positions, k=5)
   
   # Radius graph
   edge_index = graph.radius_graph(node_positions, r=1.0)
   
   # Fully connected graph
   edge_index = graph.fully_connected(num_nodes)

Node Aggregation
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Aggregate neighbor features
   aggregated = graph.aggregate_neighbors(
       node_features, 
       edge_index, 
       aggregation='mean'
   )

Graph Pooling
~~~~~~~~~~~~~

.. code-block:: python

   # Global pooling
   graph_feature = graph.global_mean_pool(node_features, batch)
   
   # Top-k pooling
   pooled, edge_index = graph.topk_pool(
       node_features, 
       edge_index, 
       k=100
   )

PageRank
~~~~~~~~

.. code-block:: python

   # Compute PageRank scores
   scores = graph.pagerank(edge_index, num_nodes)

See Also
--------

* :doc:`operators` - Tensor operations
* :doc:`physics` - Physics calculations
