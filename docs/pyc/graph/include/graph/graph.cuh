/**
 * @brief Namespace containing functions for various graph-based feature aggregation tasks.
 *
 * This namespace provides utilities to aggregate node features based on different criteria,
 * such as cluster membership, edge connectivity, or node connectivity, often utilizing
 * associated prediction values. The results are typically returned as maps where keys
 * identify the aggregation group (cluster, edge, or node) and values are the
 * corresponding aggregated feature tensors. These functions operate on PyTorch tensors.
 */
namespace graph_ {
    /**
     * @brief Aggregates node features based on unique cluster assignments.
     *
     * This function groups nodes according to their assigned cluster ID provided in `cluster_map`.
     * For each unique cluster ID found, it aggregates the features of all nodes belonging
     * to that cluster into a single representative feature tensor. The specific aggregation
     * method (e.g., sum, mean) depends on the internal implementation.
     *
     * @param cluster_map A pointer to a 1D `torch::Tensor` where each element represents the
     *                    cluster ID assigned to the corresponding node. The tensor should
     *                    contain integer-like cluster identifiers. Expected shape: `[num_nodes]`.
     * @param features A pointer to a 2D `torch::Tensor` containing the feature vectors for
     *                 each node. The i-th row corresponds to the features of the i-th node.
     *                 Expected shape: `[num_nodes, feature_dim]`.
     *
     * @return std::map<std::string, torch::Tensor> A map where:
     *         - Keys (`std::string`): String representations of the unique cluster IDs found
     *           in `cluster_map`.
     *         - Values (`torch::Tensor`): The aggregated feature tensor for all nodes belonging
     *           to the corresponding cluster. The shape of each value tensor is typically
     *           `[feature_dim]`.
     *
     * @details The function iterates through the `cluster_map`, identifies unique cluster IDs,
     *          and collects the features (`features`) associated with nodes belonging to each
     *          unique cluster. It then performs an aggregation operation (e.g., summation, averaging)
     *          on these collected features to produce a single feature vector for each cluster.
     *
     * @note The exact aggregation operation (sum, mean, max, etc.) is determined by the
     *       function's implementation and is not specified by the signature alone.
     * @note Input tensors are passed by pointer. Ensure the tensors pointed to are valid
     *       and remain valid during the function's execution.
     */
    std::map<std::string, torch::Tensor> unique_aggregation(torch::Tensor* cluster_map, torch::Tensor* features);

    /**
     * @brief Aggregates node features based on graph edges and associated edge predictions.
     *
     * This function computes an aggregated feature representation for each edge in the graph,
     * as defined by `edge_index`. The aggregation likely incorporates the features of the
     * nodes connected by the edge (`node_feature`) and potentially uses the corresponding
     * edge `prediction` value to modulate or inform the aggregation process.
     *
     * @param edge_index A pointer to a 2D `torch::Tensor` representing the graph's connectivity
     *                   in Coordinate Format (COO). Typically has shape `[2, num_edges]`, where
     *                   `edge_index[0]` contains source node indices and `edge_index[1]` contains
     *                   destination node indices for each edge.
     * @param prediction A pointer to a 1D `torch::Tensor` containing a prediction value (e.g.,
     *                   score, probability, type) associated with each edge defined in `edge_index`.
     *                   Expected shape: `[num_edges]`.
     * @param node_feature A pointer to a 2D `torch::Tensor` containing the feature vectors for
     *                     each node in the graph. Expected shape: `[num_nodes, feature_dim]`.
     *
     * @return std::map<std::string, torch::Tensor> A map where:
     *         - Keys (`std::string`): String representations identifying each edge. This might
     *           be formatted like "source_node_id-destination_node_id".
     *         - Values (`torch::Tensor`): The aggregated feature tensor computed for the
     *           corresponding edge. The shape and semantics depend on the specific aggregation
     *           logic (e.g., concatenation of node features, difference, average weighted by prediction).
     *
     * @details For each edge `(u, v)` defined in `edge_index`, the function retrieves the
     *          features of nodes `u` and `v` from `node_feature` and the edge's prediction
     *          value from `prediction`. It then applies an aggregation logic to compute a
     *          feature representation for the edge `(u, v)`.
     *
     * @note The specific method used to aggregate features for an edge (e.g., combining source
     *       and destination features, using the prediction value) is determined by the
     *       function's implementation.
     * @note The format of the string key identifying each edge is implementation-dependent.
     * @note Input tensors are passed by pointer. Ensure the tensors pointed to are valid
     *       and remain valid during the function's execution.
     */
    std::map<std::string, torch::Tensor> edge_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature);

    /**
     * @brief Aggregates node features based on graph connectivity and associated node predictions.
     *
     * This function computes an aggregated feature representation for each node in the graph.
     * The aggregation likely involves gathering features from neighboring nodes (defined by
     * `edge_index`) and potentially incorporating the node's own features (`node_feature`)
     * and its associated `prediction` value. This differs from `edge_aggregation` as the
     * focus is on deriving a new representation for each *node* based on its context.
     *
     * @param edge_index A pointer to a 2D `torch::Tensor` representing the graph's connectivity
     *                   in Coordinate Format (COO). Typically has shape `[2, num_edges]`, defining
     *                   source-destination pairs.
     * @param prediction A pointer to a 1D `torch::Tensor` containing a prediction value
     *                   associated with each *node*. Expected shape: `[num_nodes]`.
     * @param node_feature A pointer to a 2D `torch::Tensor` containing the initial feature
     *                     vectors for each node. Expected shape: `[num_nodes, feature_dim]`.
     *
     * @return std::map<std::string, torch::Tensor> A map where:
     *         - Keys (`std::string`): String representations identifying each node (e.g., "node_id").
     *         - Values (`torch::Tensor`): The aggregated feature tensor computed for the
     *           corresponding node, based on its neighborhood, original features, and prediction.
     *           The shape is typically related to `feature_dim`.
     *
     * @details For each node `i`, the function identifies its neighbors using `edge_index`.
     *          It then aggregates features (potentially from `node_feature` of neighbors and/or
     *          node `i` itself), possibly modulated by the `prediction` value for node `i` (and/or
     *          neighbors' predictions), to compute a new feature vector for node `i`. Common examples
     *          include graph convolution operations.
     *
     * @note The specific aggregation mechanism (e.g., mean/sum/max pooling of neighbor features,
     *       concatenation with self-features, weighting by predictions) is determined by the
     *       function's implementation.
     * @note The format of the string key identifying each node is implementation-dependent.
     * @note Input tensors are passed by pointer. Ensure the tensors pointed to are valid
     *       and remain valid during the function's execution.
     */
    std::map<std::string, torch::Tensor> node_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature);
}

#endif
