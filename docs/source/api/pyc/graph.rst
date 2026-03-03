Graph Operations
================

Batched graph-level operations for message-passing and graph
reconstruction, exposed via ``pyc::graph``.

``pyc::graph`` Functions
--------------------------

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``edge_aggregation(edge_index, prediction, node_feature)``
     - Aggregate node features along edges weighted by ``prediction``
   * - ``node_aggregation(edge_index, prediction, node_feature)``
     - Aggregate edge scores back to nodes
   * - ``unique_aggregation(cluster_map, features)``
     - Aggregate features by integer cluster label (reduces to number of clusters)
   * - ``PageRank(edge_index, edge_scores, alpha, threshold, norm_low, timeout, num_cls)``
     - Personalised PageRank convergence; returns cluster assignment and scores
   * - ``PageRankReconstruction(edge_index, edge_scores, pmc, …)``
     - PageRank + batched 4-momentum sum per cluster

``pyc::graph::polar`` overloads accept ``(pt, eta, phi, e)`` or a stacked
``pmu`` tensor instead of cartesian ``pmc`` for aggregation functions.

.. doxygennamespace:: pyc::graph
   :project: AnalysisG
   :members:
   :undoc-members:

Internal Kernel Namespace
--------------------------

.. doxygennamespace:: graph_
   :project: AnalysisG
   :members:
   :undoc-members:
