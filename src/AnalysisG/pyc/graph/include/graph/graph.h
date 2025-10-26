/**
 * @file graph.h
 * @brief High-performance graph aggregation operations for GNN applications
 * @defgroup pyc_graph Graph Operations
 * @ingroup module_pyc
 * @{
 *
 * This module provides efficient edge and node aggregation functions optimized
 * for Graph Neural Network (GNN) workflows in particle physics analysis.
 *
 * ## Overview
 *
 * Graph aggregation is a fundamental operation in GNNs where node or edge features
 * are combined based on network topology and classification predictions. These
 * operations are heavily used in:
 *
 * - Particle clustering (grouping constituents into jets, top quarks, etc.)
 * - Decay chain reconstruction  
 * - Object counting and multiplicity calculations
 *
 * ## Performance
 *
 * - **CUDA Accelerated**: GPU implementations for large-scale processing
 * - **Batched Operations**: Process multiple graphs simultaneously
 * - **Zero-Copy**: Direct integration with PyTorch tensors via LibTorch
 *
 * ## Usage Example
 *
 * @code{.cpp}
 * #include <graph/graph.h>
 * #include <torch/torch.h>
 *
 * // Graph topology (edge list)
 * torch::Tensor edge_index = torch::tensor({{0, 1, 2}, {1, 2, 3}});
 *
 * // Neural network predictions (which edges are "active")
 * torch::Tensor prediction = torch::tensor({1, 1, 0});
 *
 * // Node features (particle 4-momenta)
 * torch::Tensor node_features = torch::rand({4, 4});  // 4 nodes, 4 features each
 *
 * // Perform edge-based aggregation
 * auto result = graph_::edge_aggregation(&edge_index, &prediction, &node_features);
 *
 * // Access aggregated clusters
 * torch::Tensor clusters = result["clusters"];
 * torch::Tensor unique_sum = result["unique_sum"];
 * @endcode
 *
 * @see pyc_transform for coordinate transformations
 * @see pyc_physics for physics calculations
 */

#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <string>
#include <torch/torch.h>

/**
 * @namespace graph_
 * @brief Namespace containing graph aggregation functions
 */
namespace graph_ {

    /**
     * @brief Aggregate node features based on edge topology and predictions
     *
     * This function performs edge-based aggregation, combining node features
     * along edges that are classified as "active" by the neural network predictions.
     * This is commonly used for:
     * - Building composite particles from constituents
     * - Reconstructing decay chains
     * - Clustering jets
     *
     * ## Algorithm
     *
     * For each edge (i→j) where prediction > threshold:
     * 1. Aggregate features from nodes i and j
     * 2. Sum 4-momenta to form composite particles
     * 3. Track unique clusters and handle overlaps
     *
     * ## Return Dictionary Keys
     *
     * - **"clusters"**: Tensor of cluster assignments for each node
     * - **"unique_sum"**: Aggregated features for unique clusters
     * - **"reverse_sum"**: Reverse mapping from clusters to nodes
     * - **"node_sum"**: Per-node aggregated features
     *
     * @param edge_index Pointer to edge topology tensor [2 x E] where E is number of edges.
     *                   Row 0 contains source nodes, row 1 contains target nodes.
     *                   Shape: [2, num_edges], dtype: int64
     *
     * @param prediction Pointer to edge classification predictions [E].
     *                   Values typically in [0, 1] from sigmoid/softmax output.
     *                   Edges with prediction > threshold are considered active.
     *                   Shape: [num_edges], dtype: float32
     *
     * @param node_feature Pointer to node feature tensor [N x F] where N is number
     *                     of nodes and F is feature dimension (typically 4 for 4-momentum).
     *                     Shape: [num_nodes, num_features], dtype: float32
     *
     * @return std::map<std::string, torch::Tensor> Dictionary containing aggregation results:
     *         - "clusters": Node→cluster mapping
     *         - "unique_sum": Aggregated cluster features
     *         - "reverse_sum": Cluster→node reverse mapping
     *         - "node_sum": Per-node aggregated features
     *
     * @throws std::runtime_error If tensor shapes are incompatible
     * @throws std::runtime_error If CUDA is required but unavailable
     *
     * @note This function requires CUDA tensors for GPU acceleration
     * @note All input tensors must reside on the same device
     *
     * @par Performance Tip
     * For best performance, batch multiple graphs together and process simultaneously.
     *
     * @par Example
     * @code{.cpp}
     * auto result = graph_::edge_aggregation(&edges, &pred, &features);
     * torch::Tensor clusters = result["clusters"];
     * torch::Tensor aggregated = result["unique_sum"];
     *
     * // Number of reconstructed particles
     * int num_particles = aggregated.size(0);
     * @endcode
     *
     * @see node_aggregation for alternative aggregation strategy
     */
    std::map<std::string, torch::Tensor> edge_aggregation(
        torch::Tensor* edge_index,
        torch::Tensor* prediction,
        torch::Tensor* node_feature
    );

    /**
     * @brief Aggregate node features based on node classifications
     *
     * This function performs node-based aggregation, where nodes are directly
     * classified (rather than edges) and features are aggregated within clusters
     * of nodes sharing the same classification.
     *
     * This is commonly used for:
     * - Node-level particle classification
     * - Direct particle type prediction
     * - ROI (Region of Interest) aggregation
     *
     * ## Algorithm
     *
     * For each node i where prediction indicates cluster assignment:
     * 1. Group nodes by predicted cluster ID
     * 2. Aggregate features within each cluster
     * 3. Return cluster-level and node-level aggregations
     *
     * ## Return Dictionary Keys
     *
     * - **"clusters"**: Tensor of cluster assignments for each node
     * - **"unique_sum"**: Aggregated features for unique clusters
     * - **"reverse_sum"**: Reverse mapping from clusters to nodes
     * - **"node_sum"**: Per-node aggregated features
     *
     * @param edge_index Pointer to edge topology tensor [2 x E] defining graph structure.
     *                   Used to propagate features along graph edges.
     *                   Shape: [2, num_edges], dtype: int64
     *
     * @param prediction Pointer to node classification predictions [N x C] where N is
     *                   number of nodes and C is number of classes.
     *                   Typically output of softmax over node types.
     *                   Shape: [num_nodes, num_classes] or [num_nodes], dtype: float32
     *
     * @param node_feature Pointer to node feature tensor [N x F].
     *                     Contains features to aggregate (e.g., 4-momentum components).
     *                     Shape: [num_nodes, num_features], dtype: float32
     *
     * @return std::map<std::string, torch::Tensor> Dictionary with aggregation results
     *
     * @throws std::runtime_error If tensor dimensions are incompatible
     * @throws std::runtime_error If device types don't match
     *
     * @note Supports both CPU and CUDA tensors
     * @note CUDA implementation provides significant speedup for large graphs
     *
     * @par Example
     * @code{.cpp}
     * // Node predictions: class probabilities for each node
     * torch::Tensor node_pred = model->forward(node_features);
     *
     * auto result = graph_::node_aggregation(&edges, &node_pred, &features);
     *
     * // Get reconstructed particles (one per predicted class)
     * torch::Tensor particles = result["unique_sum"];
     * @endcode
     *
     * @see edge_aggregation for edge-based alternative
     */
    std::map<std::string, torch::Tensor> node_aggregation(
        torch::Tensor* edge_index,
        torch::Tensor* prediction,
        torch::Tensor* node_feature
    );
}

/** @} */ // end of pyc_graph group

#endif
