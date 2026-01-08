/**
 * @file graph.h
 * @brief This file defines the graph namespace and declares functions for edge and node aggregation.
 */

#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <string>
#include <torch/torch.h>

/**
 * @brief Namespace containing functions for graph-based data aggregation.
 * @details This namespace provides utilities specifically designed for processing graph structures,
 *          focusing on aggregating information (like node features or predictions)
 *          either onto the edges or onto the nodes themselves, based on the graph's connectivity
 *          defined by edge indices. These operations are common in Graph Neural Network (GNN)
 *          post-processing or analysis tasks.
 */
namespace graph_ {

    /**
     * @brief Performs aggregation of node features and predictions onto the graph's edges.
     * @details This function utilizes the graph's structure, defined by `edge_index`, to gather
     *          information associated with the source and target nodes of each edge. Specifically,
     *          it takes the `prediction` tensor (likely containing model outputs per node) and
     *          the `node_feature` tensor (containing input features per node) and aggregates
     *          them according to the connections specified in `edge_index`. The result is a map
     *          where keys identify the type of aggregated information (e.g., "source_prediction",
     *          "target_feature") and values are tensors containing this aggregated data, aligned
     *          with the edges. This is useful for analyzing edge properties based on connected nodes.
     *
     * @param[in] edge_index A pointer to a `torch::Tensor` representing the edge indices of the graph.
     *                       Typically, this tensor has a shape of [2, num_edges], where the first row
     *                       contains source node indices and the second row contains target node indices.
     *                       The tensor should contain integer types (e.g., `torch::kInt64`).
     *                       The pointer must not be null and must point to a valid tensor.
     * @param[in] prediction A pointer to a `torch::Tensor` containing prediction values associated with each node.
     *                       The first dimension of this tensor should correspond to the number of nodes
     *                       in the graph (i.e., shape [num_nodes, prediction_dim]).
     *                       The pointer must not be null and must point to a valid tensor.
     * @param[in] node_feature A pointer to a `torch::Tensor` containing feature vectors for each node.
     *                         The first dimension of this tensor should correspond to the number of nodes
     *                         in the graph (i.e., shape [num_nodes, feature_dim]).
     *                         The pointer must not be null and must point to a valid tensor.
     *
     * @return A `std::map<std::string, torch::Tensor>` where each key represents a specific type of
     *         aggregated information for the edges (e.g., "source_node_predictions", "target_node_features",
     *         "edge_aggregated_features"), and the corresponding `torch::Tensor` value holds the
     *         aggregated data. The tensors in the map will typically have their first dimension
     *         equal to the number of edges.
     *
     * @note The dimensions and data types of the input tensors must be compatible. Node indices
     *       in `edge_index` must be valid indices for the `prediction` and `node_feature` tensors.
     *       The function assumes the input pointers are valid and point to correctly initialized tensors.
     *       Memory management of the pointed-to tensors is the responsibility of the caller.
     */
    std::map<std::string, torch::Tensor> edge_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature);

    /**
     * @brief Performs aggregation of predictions and features onto the graph's nodes based on edge connectivity.
     * @details This function aggregates information from neighboring nodes or incident edges onto each node
     *          in the graph. Using the `edge_index` to determine connectivity, it gathers relevant
     *          `prediction` values and `node_feature` values (potentially from neighbors) and combines
     *          them to produce an aggregated representation for each node. This is a fundamental operation
     *          in message passing neural networks and graph analysis, allowing nodes to incorporate
     *          information from their local neighborhood. The specific aggregation mechanism (e.g., sum, mean, max)
     *          is determined by the function's implementation.
     *
     * @param[in] edge_index A pointer to a `torch::Tensor` representing the edge indices of the graph.
     *                       Typically, this tensor has a shape of [2, num_edges], defining source-target
     *                       node pairs for each edge. It should contain integer types (e.g., `torch::kInt64`).
     *                       The pointer must not be null and must point to a valid tensor.
     * @param[in] prediction A pointer to a `torch::Tensor` containing prediction values associated with each node.
     *                       Shape should be [num_nodes, prediction_dim]. This might represent information
     *                       to be aggregated *from* neighbors *to* a central node.
     *                       The pointer must not be null and must point to a valid tensor.
     * @param[in] node_feature A pointer to a `torch::Tensor` containing feature vectors for each node.
     *                         Shape should be [num_nodes, feature_dim]. This might represent features
     *                         to be aggregated *from* neighbors *to* a central node.
     *                         The pointer must not be null and must point to a valid tensor.
     *
     * @return A `std::map<std::string, torch::Tensor>` where keys identify the type of aggregated
     *         information per node (e.g., "aggregated_neighbor_predictions", "aggregated_neighbor_features"),
     *         and the values are `torch::Tensor` objects containing this aggregated data. The tensors
     *         in the map will typically have their first dimension equal to the number of nodes.
     *
     * @note The dimensions and data types of the input tensors must be compatible. Node indices
     *       in `edge_index` must be valid. The function assumes the input pointers point to valid tensors.
     *       The caller is responsible for managing the memory of the input tensors. The exact nature
     *       of the aggregation (e.g., which nodes' information is aggregated based on `edge_index`)
     *       depends on the function's internal logic (e.g., aggregating source node info to target nodes).
     */
    std::map<std::string, torch::Tensor> node_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature);
} // namespace graph_

#endif // GRAPH_H
