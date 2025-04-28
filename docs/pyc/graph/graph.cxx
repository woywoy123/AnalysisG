#include <graph/graph.h>
#include <utils/utils.h>

/**
 * @brief Aggregates edge information based on prediction classes and node features.
 *
 * This function takes an edge index tensor, a prediction tensor, and a node feature tensor as input.
 * It aggregates the node features based on the predicted classes for each edge, effectively summarizing
 * the features of neighboring nodes for each node in the graph, conditioned on the edge classification.
 *
 * @param edge_index A pointer to a torch::Tensor representing the edge index of the graph.
 *                   The tensor should have dimensions (2, num_edges) or (num_edges, 2), where each column
 *                   (or row, depending on the shape) represents an edge and contains the indices of the
 *                   source and destination nodes.
 * @param prediction A pointer to a torch::Tensor containing the prediction scores for each edge.
 *                   The tensor should have dimensions (num_edges, num_classes), where each row represents
 *                   an edge and contains the prediction scores for each class.
 * @param node_feature A pointer to a torch::Tensor representing the node features.
 *                     The tensor should have dimensions (num_nodes, feature_dim), where each row represents
 *                     a node and contains its feature vector.
 *
 * @return A std::map<std::string, torch::Tensor> where the keys are strings representing the class and type of output
 *         (e.g., "cls::0::node-indices", "cls::0::node-sum") and the values are the corresponding torch::Tensor.
 *         The map contains the following entries for each class:
 *         - "cls::i::node-indices": A tensor of shape (num_nodes, max_neighbors) containing the indices of the
 *           neighboring nodes for each node, based on the edges classified as class 'i'.  The values are sorted
 *           and padded with -1 to ensure consistent shape.
 *         - "cls::i::node-sum": A tensor of shape (num_nodes, feature_dim) containing the sum of the features
 *           of the neighboring nodes for each node, based on the edges classified as class 'i'.
 */
/**
 * @brief Aggregates edge predictions based on graph connectivity and node features.
 *
 * This function performs an aggregation operation over the edges of a graph. It takes the graph's
 * structure (`edge_index`), predictions associated with each edge (`prediction`), and features
 * associated with each node (`node_feature`) as input. The goal is to compute aggregated values
 * based on these inputs. The specific aggregation method (e.g., sum, mean, max pooling based on
 * source/target nodes) is determined by the internal implementation of the function.
 *
 * The aggregation might involve grouping edge predictions based on the nodes they connect to,
 * potentially using node features to weight or modify the aggregation process. The results
 * are returned as a map, allowing for multiple types of aggregations to be computed and returned
 * simultaneously, identified by string keys.
 *
 * @param edge_index A pointer to a `torch::Tensor` representing the graph's edge index.
 *                   This tensor is expected to be of shape `[2, num_edges]` and type `torch::kLong`.
 *                   The first row contains the indices of the source nodes, and the second row
 *                   contains the indices of the target nodes for each edge. It defines the
 *                   connections within the graph. The indices should correspond to the nodes
 *                   in the `node_feature` tensor.
 * @param prediction A pointer to a `torch::Tensor` containing the predictions associated with each edge.
 *                   This tensor should have a shape compatible with the number of edges, typically
 *                   `[num_edges, prediction_dimensionality]`. The data type can vary depending
 *                   on the nature of the predictions (e.g., `torch::kFloat32`). Each row corresponds
 *                   to the prediction for the edge defined at the same column index in `edge_index`.
 * @param node_feature A pointer to a `torch::Tensor` containing features for each node in the graph.
 *                     This tensor is expected to be of shape `[num_nodes, feature_dimensionality]`.
 *                     `num_nodes` should be the total number of unique nodes referenced in `edge_index`.
 *                     These features can be used to influence the aggregation process, for example,
 *                     by weighting contributions based on node properties.
 *
 * @return std::map<std::string, torch::Tensor> A map containing the results of the edge aggregation.
 *         The keys of the map are strings that identify the type of aggregation performed
 *         (e.g., "sum_aggregation_per_node", "mean_edge_features_incoming"). The values are
 *         `torch::Tensor` objects holding the computed aggregated data. The shape and data type
 *         of these result tensors depend on the specific aggregation logic implemented. For instance,
 *         an aggregation per node might result in a tensor of shape `[num_nodes, aggregated_dimensionality]`.
 *
 * @note The function receives pointers to tensors. It does not take ownership of these tensors.
 *       The caller is responsible for ensuring the tensors remain valid for the duration of the function call.
 * @warning Input tensors must be valid and correctly formatted. `edge_index` values must be within the valid
 *          range of node indices `[0, num_nodes - 1]`. The number of edges implied by `edge_index`
 *          must match the first dimension of the `prediction` tensor. Failure to meet these requirements
 *          may lead to runtime errors (e.g., out-of-bounds access) or incorrect aggregation results.
 *          Ensure tensors reside on the same device (CPU or specific GPU) for compatibility.
 */
std::map<std::string, torch::Tensor> graph_::edge_aggregation(
    torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature
);

/**
 * @brief Aggregates node features based on edge indices and predictions.
 * 
 * This function performs node aggregation by processing the given edge indices,
 * predictions, and node features. It ensures the edge indices are in the correct
 * format and uses them to index predictions. The function then delegates the
 * aggregation process to `graph_::edge_aggregation`.
 * 
 * @param edge_index Pointer to a tensor containing edge indices. It is expected
 *        to have dimensions [2, num_edges] or [num_edges, 2].
 * @param prediction Pointer to a tensor containing predictions for each edge.
 * @param node_feature Pointer to a tensor containing node features.
 * 
 * @return A map where the keys are strings representing aggregation results
 *         and the values are tensors containing the aggregated data.
 */
std::map<std::string, torch::Tensor> graph_::node_aggregation(
    torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature
);
