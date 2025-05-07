/**
 * @file graph.cuh
 * @brief Provides high-level graph operations and utilities for CUDA.
 */

#ifndef GRAPH_CUH
#define GRAPH_CUH

#include <torch/torch.h> ///< Includes PyTorch C++ headers for tensor operations.
#include <utils/utils.cuh> ///< Includes utility functions for CUDA operations.
#include <graph/base.cuh> ///< Includes base graph utilities.
#include <graph/pagerank.cuh> ///< Includes PageRank algorithm definitions.
#include <graph/reconstruction.cuh> ///< Includes graph reconstruction utilities.

/**
 * @brief Namespace for graph-related operations.
 */
namespace graph_ {

/**
 * @brief Aggregates edge features based on edge indices and predictions.
 *
 * @param edge_index Input tensor containing edge indices of the graph.
 * @param prediction Input tensor containing edge predictions.
 * @param node_feature Input tensor containing node features.
 * @return A map of strings to tensors containing aggregated edge features.
 */
std::map<std::string, torch::Tensor> edge_aggregation(
    torch::Tensor edge_index,
    torch::Tensor prediction,
    torch::Tensor node_feature
);

/**
 * @brief Aggregates node features based on edge indices and predictions.
 *
 * @param edge_index Input tensor containing edge indices of the graph.
 * @param prediction Input tensor containing edge predictions.
 * @param node_feature Input tensor containing node features.
 * @return A map of strings to tensors containing aggregated node features.
 */
std::map<std::string, torch::Tensor> node_aggregation(
    torch::Tensor edge_index,
    torch::Tensor prediction,
    torch::Tensor node_feature
);

/**
 * @brief Aggregates unique features based on cluster mapping and predictions.
 *
 * @param cluster_map Input tensor mapping nodes to clusters.
 * @param features Input tensor containing node features.
 * @param prediction Input tensor containing edge predictions.
 * @return A map of strings to tensors containing aggregated unique features.
 */
std::map<std::string, torch::Tensor> unique_aggregation(
    torch::Tensor cluster_map,
    torch::Tensor features,
    torch::Tensor prediction
);

/**
 * @brief Reconstructs a graph based on edge indices, truth edges, and predictions.
 *
 * @param edge_index Input tensor containing edge indices of the graph.
 * @param truth_edge Input tensor containing ground truth edges.
 * @param pred_edge Input tensor containing predicted edges.
 * @param node_feature Input tensor containing node features.
 * @param masses Input tensor containing node masses.
 * @param trk_data Input tensor containing track data.
 * @return A map of strings to tensors containing the reconstructed graph.
 */
std::map<std::string, torch::Tensor> reconstruction(
    torch::Tensor edge_index,
    torch::Tensor truth_edge,
    torch::Tensor pred_edge,
    torch::Tensor node_feature,
    torch::Tensor masses,
    torch::Tensor trk_data
);

} // namespace graph_

#endif // GRAPH_CUH
