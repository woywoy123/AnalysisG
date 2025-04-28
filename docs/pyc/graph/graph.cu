/**
 * @file graph.cu
 * @brief Implements CUDA-accelerated graph aggregation functions for node and edge features using PyTorch tensors.
 *
 * This file provides implementations for various graph aggregation strategies commonly used in graph neural networks
 * and graph analysis tasks. The functions operate on PyTorch tensors and leverage CUDA for performance.
 * Key functionalities include aggregating node features based on cluster assignments and aggregating node features
 * along edges based on prediction scores.
 */

/**
 * @brief Aggregates node features based on unique cluster assignments provided in a cluster map.
 *
 * This function performs a unique aggregation operation. It identifies unique cluster identifiers from the
 * `cluster_map` and sums the features of all nodes belonging to the same cluster. This is useful for
 * tasks like graph pooling or summarizing information within identified communities or segments.
 *
 * The aggregation process involves:
 * 1. Identifying all unique cluster IDs present in the `cluster_map`.
 * 2. For each unique cluster ID, iterating through the `cluster_map` to find all nodes assigned to that cluster.
 * 3. Summing the corresponding feature vectors (rows) from the `features` tensor for all nodes belonging to the current cluster.
 * 4. Storing the resulting aggregated feature vector.
 *
 * The function returns a map where:
 * - The key "unique" maps to a tensor containing the sorted unique cluster IDs found.
 * - The key "node-sum" maps to a tensor where each row corresponds to the aggregated feature vector for the unique cluster ID at the same index in the "unique" tensor.
 *
 * @param cluster_map A pointer to a 2D PyTorch tensor (`torch::Tensor`) representing the cluster assignments for each node.
 *                    Expected dimensions: `(n_nodes, ij_node)`, where `n_nodes` is the total number of nodes in the graph,
 *                    and `ij_node` represents the number of potential cluster assignments per node (often 1 if each node belongs to a single cluster).
 *                    The tensor should contain integer cluster identifiers.
 * @param features A pointer to a 2D PyTorch tensor (`torch::Tensor`) containing the feature vectors for each node.
 *                 Expected dimensions: `(n_nodes, n_feat)`, where `n_nodes` is the total number of nodes (matching the first dimension of `cluster_map`),
 *                 and `n_feat` is the dimensionality of the node features.
 *                 The tensor should contain numerical feature values (e.g., float or double).
 * @return A `std::map<std::string, torch::Tensor>` containing the results:
 *         - `"unique"`: A 1D tensor containing the sorted unique cluster identifiers found in `cluster_map`.
 *         - `"node-sum"`: A 2D tensor where each row `i` contains the sum of feature vectors of all nodes belonging to the cluster identifier `unique[i]`. The dimensions will be `(num_unique_clusters, n_feat)`.
 */
std::map<std::string, torch::Tensor> graph_::unique_aggregation(torch::Tensor* cluster_map, torch::Tensor* features);

/**
 * @brief Aggregates node features along edges, weighted or filtered by edge-specific prediction scores.
 *
 * This function performs edge-based aggregation. For each prediction class or score type provided in the `prediction` tensor,
 * it aggregates features of nodes connected by edges. Typically, it sums the features of the source nodes (`edge_index[0]`)
 * for each target node (`edge_index[1]`), potentially filtered or weighted by the corresponding prediction scores.
 * This is a core operation in message passing neural networks.
 *
 * The aggregation process for each prediction class `x`:
 * 1. Iterate through each edge `e` defined in `edge_index`. Let the edge connect node `u` (source) to node `v` (target).
 * 2. Obtain the prediction score(s) for edge `e` from the `prediction` tensor (column `x`).
 * 3. Based on the prediction score (e.g., if it exceeds a threshold or using it as a weight), aggregate the feature vector of the source node `u` (`node_feature[u]`) towards the target node `v`. The exact aggregation mechanism (sum, mean, max, weighted sum) depends on the specific implementation details not shown in the declaration but implied by the "node-sum" key. This documentation assumes summation.
 * 4. Keep track of which source nodes contributed to the aggregation for each target node.
 *
 * The function returns a map containing separate results for each prediction class (column in the `prediction` tensor).
 *
 * @param edge_index A pointer to a 2D PyTorch tensor (`torch::Tensor`) representing the graph's edge connectivity.
 *                   Expected dimensions: `(2, num_edges)` (standard PyG format) or `(num_edges, 2)`.
 *                   - If `(2, num_edges)`: `edge_index[0]` contains source node indices, `edge_index[1]` contains target node indices.
 *                   - If `(num_edges, 2)`: `edge_index[i, 0]` is source, `edge_index[i, 1]` is target for edge `i`.
 *                   Node indices should be integer types.
 * @param prediction A pointer to a PyTorch tensor (`torch::Tensor`) containing prediction scores associated with each edge.
 *                   Expected dimensions: `(num_edges, pred_lx)`, where `num_edges` matches the number of edges derived from `edge_index`,
 *                   and `pred_lx` is the number of different prediction scores or classes associated with each edge.
 *                   Values are typically probabilities, logits, or binary indicators.
 * @param node_feature A pointer to a 2D PyTorch tensor (`torch::Tensor`) containing the feature vectors for each node.
 *                     Expected dimensions: `(node_lx, node_fx)`, where `node_lx` is the total number of nodes in the graph,
 *                     and `node_fx` is the dimensionality of the node features.
 * @return A `std::map<std::string, torch::Tensor>` containing aggregated results for each prediction class `x` (from 0 to `pred_lx - 1`):
 *         - `"cls::x::node-indices"`: A tensor containing the indices of the source nodes that contributed features to the aggregation for each target node, corresponding to prediction class `x`. The exact structure depends on the implementation (e.g., could be a flattened list or part of a COO representation).
 *         - `"cls::x::node-sum"`: A tensor containing the aggregated (summed) node features for each target node, based on the contributions determined by prediction class `x`. The dimensions might be `(node_lx, node_fx)` if aggregation is done per target node, or potentially different based on the specific aggregation strategy.
 */
std::map<std::string, torch::Tensor> graph_::edge_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature);

/**
 * @brief Aggregates node features along edges using node-level prediction scores.
 *
 * This function adapts the edge aggregation mechanism for scenarios where prediction scores are initially available
 * at the node level rather than the edge level. It first "lifts" the node predictions to the edges by indexing
 * the `prediction` tensor using the source node indices from `edge_index`. The resulting edge-specific predictions
 * are then used to perform the aggregation via the `edge_aggregation` function.
 *
 * The process involves:
 * 1. Extracting source node indices from `edge_index` (e.g., `edge_index[0]`).
 * 2. Using these indices to gather the corresponding prediction scores from the node-level `prediction` tensor. This creates an edge-level prediction tensor.
 *    `edge_predictions[e] = prediction[edge_index[0, e]]` (assuming `(2, num_edges)` format).
 * 3. Calling the `edge_aggregation` function with the original `edge_index`, the newly created `edge_predictions`, and the `node_feature` tensor.
 *
 * This is useful when a model predicts node properties, and these properties need to influence how information flows along edges.
 *
 * @param edge_index A pointer to a 2D PyTorch tensor (`torch::Tensor`) representing the graph's edge connectivity.
 *                   Expected dimensions: `(2, num_edges)` or `(num_edges, 2)`. See `edge_aggregation` for details.
 *                   Node indices should be integer types.
 * @param prediction A pointer to a 1D or 2D PyTorch tensor (`torch::Tensor`) representing prediction scores associated with each *node*.
 *                   - If 1D: Expected dimensions `(node_lx,)`, where `node_lx` is the number of nodes. Assumes a single prediction score per node.
 *                   - If 2D: Expected dimensions `(node_lx, pred_lx)`, where `pred_lx` is the number of prediction scores/classes per node.
 *                   The function will index this tensor using source node indices from `edge_index`.
 * @param node_feature A pointer to a 2D PyTorch tensor (`torch::Tensor`) containing the feature vectors for each node.
 *                     Expected dimensions: `(node_lx, node_fx)`, where `node_lx` is the total number of nodes, and `node_fx` is the feature dimensionality.
 * @return A `std::map<std::string, torch::Tensor>` identical in structure to the return value of `edge_aggregation`. It contains the aggregated node features and contributing node indices for each prediction class, derived from the initial node-level predictions.
 *         Keys will be `"cls::x::node-indices"` and `"cls::x::node-sum"`.
 */
std::map<std::string, torch::Tensor> graph_::node_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature);
