/**
 * @brief Computes the PageRank scores for nodes in a graph.
 * @details This function calculates the PageRank scores based on the provided graph structure
 * (edge_index) and optional edge weights (edge_scores). It iteratively updates node ranks
 * based on the link structure and a damping factor (alpha). The process continues until
 * the change in ranks falls below a specified threshold or a timeout is reached.
 * The function handles potential transposition of input tensors if their dimensions
 * are not in the expected format (2xN). The actual computation is delegated to an
 * underlying implementation, potentially utilizing CUDA if available (`PYC_CUDA` is defined
 * during compilation). Input tensors are automatically moved to the correct device.
 *
 * @param edge_index A tensor representing the graph's edges. Expected shape is [2, num_edges],
 *                   where edge_index[0] contains source nodes and edge_index[1] contains
 *                   destination nodes. If the input shape is [num_edges, 2], it will be
 *                   transposed internally. Must be on the correct computation device (CPU/CUDA).
 * @param edge_scores A tensor representing the scores or weights associated with each edge.
 *                    Expected shape is [num_edges]. This tensor influences the probability
 *                    of traversing an edge during the random walk simulation underlying PageRank.
 *                    Must be on the correct computation device (CPU/CUDA).
 * @param alpha The damping factor for the PageRank algorithm (typically around 0.85).
 *              Represents the probability at each step that the random surfer will continue
 *              following links, versus teleporting to a random node.
 * @param threshold The convergence threshold. Iterations stop when the maximum absolute change
 *                  in PageRank scores between consecutive iterations falls below this value.
 * @param norm_low A lower bound used during normalization steps within the algorithm,
 *                 potentially to avoid division by zero or handle specific normalization schemes.
 * @param timeout A timeout limit (in implementation-defined units, e.g., milliseconds or iterations)
 *                for the PageRank computation. If the computation exceeds this limit, it may terminate
 *                before full convergence.
 * @param num_cls The number of classes or clusters. This parameter might be used for personalized
 *                PageRank variants or related algorithms where node importance is calculated
 *                with respect to specific classes or topics. Its exact usage depends on the
 *                underlying implementation.
 *
 * @return A torch::Dict<std::string, torch::Tensor> containing the results of the PageRank
 *         computation. The specific keys and tensor contents depend on the underlying
 *         `graph_::page_rank` implementation. Typically, it includes a tensor named "ranks"
 *         containing the calculated PageRank score for each node.
 *
 * @note The function ensures input tensors `edge_index` and `edge_scores` are moved to the
 *       appropriate device using `changedev` before computation.
 * @note The core computation relies on the `graph_::page_rank` function, which may have
 *       CPU and CUDA implementations selected based on the `PYC_CUDA` macro.
 * @note Input tensor `edge_index` might be transposed internally if its shape is detected
 *       as [N, 2] instead of the expected [2, N].
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::PageRank(
    torch::Tensor edge_index, torch::Tensor edge_scores,
    double alpha, double threshold, double norm_low, long timeout, long num_cls
);


/**
 * @brief Performs graph reconstruction or refinement using a PageRank-like iterative process.
 * @details This function utilizes edge scores and Personalized Markov Chain (PMC) information
 * alongside the graph structure (edge_index) to reconstruct or refine graph properties.
 * It applies softmax normalization to the input `edge_scores` and then runs an iterative
 * algorithm similar to PageRank, potentially incorporating the PMC data to personalize or
 * guide the process. The function handles tensor device placement and ensures the `edge_index`
 * tensor has the shape [2, N]. The core computation might be offloaded to CUDA if available
 * (`PYC_CUDA` macro defined during compilation).
 *
 * @param edge_index A tensor representing the graph's edge index. Expected shape is [2, num_edges]
 *                   or [num_edges, 2]. It will be transposed to [2, num_edges] if necessary.
 *                   Must be on the correct computation device (CPU/CUDA).
 * @param edge_scores A tensor containing scores associated with each edge. Softmax normalization
 *                    is applied to this tensor internally before use in the algorithm.
 *                    Expected shape is compatible with `edge_index`, typically [num_edges].
 *                    Must be on the correct computation device (CPU/CUDA).
 * @param pmc A tensor representing Personalized Markov Chain information. This likely contains
 *            node-specific data used to influence the PageRank calculation, potentially biasing
 *            the random walks or teleportation probabilities. Must be on the correct
 *            computation device (CPU/CUDA).
 * @param alpha The damping factor for the PageRank-like algorithm (e.g., 0.85). Controls the
 *              balance between following existing edges (influenced by `edge_scores`) and
 *              teleporting (potentially influenced by `pmc`).
 * @param threshold A threshold value used during the reconstruction process. This could be a
 *                  convergence threshold for the iterative algorithm or a filtering threshold
 *                  applied to edges or nodes based on their scores.
 * @param norm_low A lower bound used for normalization purposes within the algorithm, possibly
 *                 to prevent numerical instability.
 * @param timeout A timeout value (in implementation-defined units, e.g., milliseconds or iterations)
 *                to limit the computation time. The process might stop before full convergence
 *                if the timeout is reached.
 * @param num_cls The number of classes or clusters. This parameter might influence how the `pmc`
 *                data is used or affect the reconstruction logic, potentially aiming to
 *                reconstruct class-specific graph structures.
 *
 * @return A torch::Dict<std::string, torch::Tensor> containing the results of the reconstruction.
 *         The specific keys and tensor values depend on the underlying `graph_::page_rank_reconstruction`
 *         implementation (potentially CUDA-accelerated). It might contain reconstructed edge scores,
 *         node rankings, or other graph properties. The dictionary is converted from an internal
 *         std::map representation.
 *
 * @note The function automatically handles moving input tensors (`edge_index`, `edge_scores`, `pmc`)
 *       to the appropriate computation device (CPU/CUDA) via `changedev`.
 * @note The core computation relies on the `graph_::page_rank_reconstruction` function, which requires
 *       a CUDA implementation if the `PYC_CUDA` macro is defined during compilation.
 * @note Input `edge_scores` tensor will be modified internally by applying softmax normalization.
 * @note Input `edge_index` might be transposed internally if its shape is detected as [N, 2].
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::PageRankReconstruction(
    torch::Tensor edge_index, torch::Tensor edge_scores, torch::Tensor pmc,
    double alpha, double threshold, double norm_low, long timeout, long num_cls
);


/**
 * @brief Aggregates features associated with edges based on graph connectivity and prediction scores.
 * @details This function performs aggregation operations (like sum, mean, max, etc.) on edge features.
 * It uses the `edge_index` tensor to define the graph structure (which nodes are connected by edges)
 * and the `prediction` tensor, likely containing weights or scores for each edge, to potentially
 * modulate the aggregation. The `edge_feature` tensor provides the actual feature vectors associated
 * with each edge that need to be aggregated, typically towards the connected nodes. The specific
 * aggregation methods performed depend on the underlying implementation (`graph_::edge_aggregation`).
 * Input tensors are moved to the appropriate device.
 *
 * @param edge_index A tensor of shape [2, num_edges] containing source and target node indices for each edge.
 *                   Defines the graph connectivity used for aggregation. Must be on the correct device.
 * @param prediction A tensor containing prediction scores or weights associated with each edge. Shape is
 *                   typically [num_edges] or [num_edges, num_prediction_features]. Used to potentially
 *                   weight the features during aggregation. Must be on the correct device.
 * @param edge_feature A tensor containing the features associated with each edge that are to be aggregated.
 *                     Shape is typically [num_edges, num_edge_features]. Must be on the correct device.
 *
 * @return A torch::Dict<std::string, torch::Tensor> where keys are strings identifying the aggregation
 *         type (e.g., "sum", "mean", "max") and values are tensors containing the aggregated features.
 *         The shape and interpretation of the output tensors depend on the specific aggregation strategy
 *         (e.g., aggregating edge features onto nodes).
 *
 * @note Input tensors are managed for device placement via `changedev`.
 * @note The underlying implementation `graph_::edge_aggregation` determines the exact aggregation operations.
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor edge_feature
);

/**
 * @brief Aggregates features associated with nodes based on graph connectivity and edge predictions.
 * @details This function performs aggregation operations (like sum, mean, max, etc.) using node features.
 * It uses the `edge_index` tensor to determine which nodes influence each other based on graph connectivity.
 * The `prediction` tensor, likely associated with edges, might act as weights or gates for the aggregation
 * process (e.g., weighting the contribution of a neighbor's features based on the edge prediction score).
 * The `node_feature` tensor provides the feature vectors associated with each node, which are then aggregated
 * according to the graph structure and edge predictions. The specific aggregation methods depend on the
 * underlying implementation (`graph_::node_aggregation`). Input tensors are moved to the appropriate device.
 *
 * @param edge_index A tensor of shape [2, num_edges] containing source and target node indices for each edge.
 *                   Defines the graph connectivity used for aggregation (e.g., which nodes aggregate features from which neighbors).
 *                   Must be on the correct device.
 * @param prediction A tensor containing prediction scores or weights, likely associated with each edge. Shape could be
 *                   [num_edges] or [num_edges, num_prediction_features]. Used to potentially modulate the aggregation
 *                   of neighbor features. Must be on the correct device.
 * @param node_feature A tensor containing the features associated with each node. Shape is typically
 *                     [num_nodes, num_node_features]. These are the features being aggregated. Must be on the correct device.
 *
 * @return A torch::Dict<std::string, torch::Tensor> where keys are strings identifying the aggregation
 *         type (e.g., "sum", "mean", "max") and values are tensors containing the aggregated features for each node.
 *         The shape of the output tensors will typically be [num_nodes, aggregated_feature_size].
 *
 * @note Input tensors are managed for device placement via `changedev`.
 * @note The underlying implementation `graph_::node_aggregation` determines the exact aggregation operations.
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature
);

/**
 * @brief Aggregates features based on a cluster mapping.
 * @details This function performs aggregation operations (like sum, mean, max, etc.) on a set of features,
 * grouping them according to a provided `cluster_map`. Each element in the `feature` tensor is assigned
 * to a cluster specified by the corresponding element in `cluster_map`. Features within the same cluster
 * are then aggregated together. This is useful for pooling features belonging to the same super-node or cluster.
 * The specific aggregation methods depend on the underlying implementation (`graph_::unique_aggregation`).
 * Input tensors are moved to the appropriate device.
 *
 * @param cluster_map A tensor, typically 1D, where `cluster_map[i]` indicates the cluster ID for the i-th feature vector.
 *                    Must be on the correct device. The number of elements should match the first dimension of `feature`.
 * @param feature A tensor containing the features to be aggregated. Shape is typically [num_elements, num_features].
 *                Must be on the correct device.
 *
 * @return A torch::Dict<std::string, torch::Tensor> where keys are strings identifying the aggregation
 *         type (e.g., "sum", "mean", "max") and values are tensors containing the aggregated features for each unique cluster.
 *         The shape of the output tensors will typically be [num_clusters, num_features].
 *
 * @note Input tensors are managed for device placement via `changedev`.
 * @note The underlying implementation `graph_::unique_aggregation` determines the exact aggregation operations.
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::unique_aggregation(
        torch::Tensor cluster_map, torch::Tensor feature
);

/**
 * @brief Aggregates edge features using polar coordinate inputs, potentially after internal conversion.
 * @details This function takes edge indices, prediction scores, and particle kinematic data (`pmu`) as input.
 * It assumes `pmu` contains kinematic information (potentially in polar coordinates like Pt, Eta, Phi, E,
 * although the documentation mentions transforming to Cartesian Px, Py, Pz, E internally). It then performs
 * edge-based aggregation using the graph structure (`edge_index`), edge scores (`prediction`), and the
 * (potentially transformed) kinematic data. The result is a dictionary of aggregated features.
 *
 * @param edge_index A tensor of shape [2, num_edges] containing source and target node indices for each edge.
 *                   Must be on the correct device.
 * @param prediction A tensor containing prediction scores or weights associated with each edge. Shape is
 *                   typically [num_edges] or [num_edges, num_prediction_features]. Must be on the correct device.
 * @param pmu A tensor containing kinematic data for each particle/node, potentially in polar coordinates.
 *            This data is transformed internally (likely to Cartesian Px, Py, Pz, E) before being used
 *            in the aggregation. Shape is typically [num_nodes, num_kinematic_features]. Must be on the correct device.
 *
 * @return A torch::Dict<std::string, torch::Tensor> containing the aggregated edge features, keyed by aggregation type (e.g., "sum", "mean").
 *
 * @note This function exists within the `pyc::graph::polar` namespace, suggesting an intended focus on polar coordinates,
 *       but the documentation indicates an internal transformation to Cartesian coordinates derived from `pmu`.
 * @note Input tensors are managed for device placement via `changedev`.
 * @note Relies on the underlying `graph_::edge_aggregation` implementation after coordinate transformation.
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::polar::edge_aggregation(
    torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
);

/**
 * @brief Aggregates node features using polar coordinate inputs, potentially after internal conversion.
 * @details This function takes edge indices, prediction scores (likely edge-based), and particle kinematic data (`pmu`) as input.
 * It assumes `pmu` contains kinematic information for nodes (potentially in polar coordinates like Pt, Eta, Phi, E).
 * This kinematic data is likely transformed internally (e.g., to Cartesian Px, Py, Pz, E) and used as node features.
 * The function then performs node-based aggregation, where node features are aggregated based on graph connectivity (`edge_index`)
 * and potentially modulated by edge predictions (`prediction`). The result is a dictionary of aggregated node features.
 *
 * @param edge_index A tensor of shape [2, num_edges] containing source and target node indices for each edge.
 *                   Defines the graph connectivity for aggregation. Must be on the correct device.
 * @param prediction A tensor containing prediction scores or weights, likely associated with edges, used to modulate aggregation.
 *                   Shape is typically [num_edges] or [num_edges, num_prediction_features]. Must be on the correct device.
 * @param pmu A tensor containing kinematic data for each particle/node, potentially in polar coordinates.
 *            This data is transformed internally (likely to Cartesian Px, Py, Pz, E) and serves as the node features
 *            to be aggregated. Shape is typically [num_nodes, num_kinematic_features]. Must be on the correct device.
 *
 * @return A torch::Dict<std::string, torch::Tensor> containing the aggregated node features, keyed by aggregation type (e.g., "sum", "mean").
 *         Output tensors typically have shape [num_nodes, aggregated_feature_size].
 *
 * @note This function exists within the `pyc::graph::polar` namespace. It likely uses the `pmu` tensor as the source
 *       for node features, potentially after an internal coordinate transformation.
 * @note Input tensors are managed for device placement via `changedev`.
 * @note Relies on the underlying `graph_::node_aggregation` implementation after coordinate transformation.
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::polar::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
);

// NOTE: The documentation for the following function `pyc::graph::polar::edge_aggregation`
//       was duplicated multiple times in the original source. The consolidated and refined
//       documentation is provided below.

/**
 * @brief Aggregates features along the edges of a graph using polar coordinate inputs for node kinematics.
 * @details This function performs edge-centric aggregation on a graph. It takes the graph structure (`edge_index`),
 * edge-level predictions or scores (`prediction`), and node kinematic variables provided in polar coordinates
 * (`pt`, `eta`, `phi`, `e`). Internally, it first transforms the polar kinematic variables into Cartesian
 * momentum components (Px, Py, Pz, E). Then, it uses an underlying edge aggregation mechanism
 * (`graph_::edge_aggregation`) that likely combines the edge predictions and the derived Cartesian momentum
 * features based on the graph connectivity. The results of different aggregation types (e.g., sum, mean)
 * are returned in a dictionary.
 *
 * @param edge_index A tensor of shape [2, num_edges] containing source and target node indices for each edge.
 *                   Defines the graph connectivity. May be moved to the appropriate device internally.
 * @param prediction A tensor containing scores or features associated with each edge. Shape is typically
 *                   [num_edges] or [num_edges, num_prediction_features]. Must be on the correct device.
 * @param pt A 1D tensor of length num_nodes representing the transverse momentum (pT) of each node. Must be on the correct device.
 * @param eta A 1D tensor of length num_nodes containing the pseudorapidity (η) values for each node. Must be on the correct device.
 * @param phi A 1D tensor of length num_nodes containing the azimuthal angle (φ) values for each node. Must be on the correct device.
 * @param e A 1D tensor of length num_nodes representing the energy (E) of each node. Must be on the correct device.
 *
 * @return A torch::Dict<std::string, torch::Tensor> where keys are strings identifying the aggregation
 *         type (e.g., "sum", "mean") and values are tensors containing the corresponding aggregated features.
 *         The exact nature and shape of the output tensors depend on the underlying aggregation implementation.
 *
 * @note This function resides in the `pyc::graph::polar` namespace, indicating the input format for kinematics.
 * @note An internal transformation from polar (pt, eta, phi, e) to Cartesian (Px, Py, Pz, E) coordinates occurs before aggregation.
 * @note Input tensors `pt`, `eta`, `phi`, `e`, and `prediction` must be on the correct device; `edge_index` might be moved internally.
 * @note The core aggregation logic is handled by `graph_::edge_aggregation`.
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::polar::edge_aggregation(
    torch::Tensor edge_index, torch::Tensor prediction,
    torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
);


/**
 * @brief Aggregates node features based on graph connectivity and predictions, using polar coordinate inputs for kinematics.
 * @details This function performs node-centric aggregation. It takes the graph structure (`edge_index`),
 * edge-level predictions or scores (`prediction`), and node kinematic variables provided in polar coordinates
 * (`pt`, `eta`, `phi`, `e`). Internally, it first transforms the polar kinematic variables into Cartesian
 * momentum components (Px, Py, Pz, E), which likely serve as the base node features. Then, it uses an
 * underlying node aggregation mechanism (`graph_::node_aggregation`) that aggregates features from neighboring
 * nodes based on the graph connectivity, potentially weighting the contributions using the `prediction` scores
 * associated with the connecting edges. The results of different aggregation types (e.g., sum, mean)
 * are returned in a dictionary.
 *
 * @param edge_index A tensor of shape [2, num_edges] representing the edge indices of the graph (source and target nodes).
 *                   Defines neighborhood relationships for aggregation. Must be on the correct device.
 * @param prediction A tensor containing prediction scores or weights, likely associated with each edge. Used to modulate
 *                   the aggregation process. Shape typically [num_edges] or [num_edges, num_prediction_features]. Must be on the correct device.
 * @param pt A 1D tensor of length num_nodes containing the transverse momentum (pt) of each node. Must be on the correct device.
 * @param eta A 1D tensor of length num_nodes containing the pseudorapidity (eta) of each node. Must be on the correct device.
 * @param phi A 1D tensor of length num_nodes containing the azimuthal angle (phi) of each node. Must be on the correct device.
 * @param e A 1D tensor of length num_nodes containing the energy (e) of each node. Must be on the correct device.
 *
 * @return A torch::Dict<std::string, torch::Tensor> containing the aggregated node features, keyed by aggregation type
 *         (e.g., "sum", "mean"). Output tensors typically have shape [num_nodes, aggregated_feature_size].
 *
 * @note This function resides in the `pyc::graph::polar` namespace, indicating the input format for kinematics.
 * @note An internal transformation from polar (pt, eta, phi, e) to Cartesian (Px, Py, Pz, E) coordinates occurs,
 *       and the result is used as the node features for aggregation.
 * @note Input tensors are managed for device placement via `changedev`.
 * @note The core aggregation logic is handled by `graph_::node_aggregation`.
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::polar::node_aggregation(
    torch::Tensor edge_index, torch::Tensor prediction,
    torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
);

/**
 * @brief Aggregates edge features using Cartesian coordinate inputs represented by a combined tensor.
 * @details This function performs edge-centric aggregation using pre-computed Cartesian kinematic data.
 * It takes the graph structure (`edge_index`), edge-level predictions or scores (`prediction`), and
 * node kinematic data combined into a single tensor `pmc` (presumably containing Px, Py, Pz, E for each node).
 * It uses an underlying edge aggregation mechanism (`graph_::edge_aggregation`) that likely combines
 * the edge predictions and the node kinematic features based on the graph connectivity. The results of
 * different aggregation types (e.g., sum, mean) are returned in a dictionary.
 *
 * @param edge_index A tensor of shape [2, num_edges] containing source and target node indices for each edge.
 *                   Must be on the correct device.
 * @param prediction A tensor containing prediction scores or weights associated with each edge. Shape is
 *                   typically [num_edges] or [num_edges, num_prediction_features]. Must be on the correct device.
 * @param pmc A tensor containing Cartesian kinematic data (likely Px, Py, Pz, E) for each node.
 *            Shape is typically [num_nodes, 4]. This data is used in the aggregation process. Must be on the correct device.
 *
 * @return A torch::Dict<std::string, torch::Tensor> containing the aggregated edge features, keyed by aggregation type (e.g., "sum", "mean").
 *
 * @note This function resides in the `pyc::graph::cartesian` namespace, indicating the input format for kinematics.
 * @note Input tensors are managed for device placement via `changedev`.
 * @note Relies on the underlying `graph_::edge_aggregation` implementation.
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
);

/**
 * @brief Aggregates node features using Cartesian coordinate inputs represented by a combined tensor.
 * @details This function performs node-centric aggregation using pre-computed Cartesian kinematic data.
 * It takes the graph structure (`edge_index`), edge-level predictions or scores (`prediction`), and
 * node kinematic data combined into a single tensor `pmc` (presumably containing Px, Py, Pz, E for each node),
 * which serves as the base node features. It uses an underlying node aggregation mechanism (`graph_::node_aggregation`)
 * that aggregates features from neighboring nodes based on the graph connectivity, potentially weighting contributions
 * using the `prediction` scores. The results of different aggregation types (e.g., sum, mean) are returned in a dictionary.
 *
 * @param edge_index A tensor of shape [2, num_edges] containing source and target node indices for each edge.
 *                   Defines neighborhood relationships. Must be on the correct device.
 * @param prediction A tensor containing prediction scores or weights, likely associated with edges, used to modulate aggregation.
 *                   Shape typically [num_edges] or [num_edges, num_prediction_features]. Must be on the correct device.
 * @param pmc A tensor containing Cartesian kinematic data (likely Px, Py, Pz, E) for each node.
 *            Shape is typically [num_nodes, 4]. This serves as the node features to be aggregated. Must be on the correct device.
 *
 * @return A torch::Dict<std::string, torch::Tensor> containing the aggregated node features, keyed by aggregation type (e.g., "sum", "mean").
 *         Output tensors typically have shape [num_nodes, aggregated_feature_size].
 *
 * @note This function resides in the `pyc::graph::cartesian` namespace, indicating the input format for kinematics.
 * @note Input tensors are managed for device placement via `changedev`.
 * @note Relies on the underlying `graph_::node_aggregation` implementation.
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
);

/**
 * @brief Performs edge aggregation for a graph using explicitly provided Cartesian coordinate components.
 * @details This function aggregates features along the edges of a graph. It takes the graph structure (`edge_index`),
 * edge-level predictions or scores (`prediction`), and node kinematic variables provided as separate Cartesian
 * components (`px`, `py`, `pz`, `e`). It uses an underlying edge aggregation mechanism (`graph_::edge_aggregation`)
 * that combines the edge predictions and the node kinematic features based on the graph connectivity. The results
 * of different aggregation types (e.g., sum, mean) are returned in a dictionary.
 *
 * @param edge_index A tensor of shape [2, num_edges] containing source and target node indices for each edge.
 *                   Must be on the correct device.
 * @param prediction A tensor containing prediction scores or weights associated with each edge. Shape is
 *                   typically [num_edges] or [num_edges, num_prediction_features]. Must be on the correct device.
 * @param px A 1D tensor of length num_nodes representing the x-component of momentum for each node. Must be on the correct device.
 * @param py A 1D tensor of length num_nodes representing the y-component of momentum for each node. Must be on the correct device.
 * @param pz A 1D tensor of length num_nodes representing the z-component of momentum for each node. Must be on the correct device.
 * @param e A 1D tensor of length num_nodes representing the energy of each node. Must be on the correct device.
 *
 * @return A torch::Dict<std::string, torch::Tensor> containing the aggregated edge features, keyed by aggregation type
 *         (e.g., "sum", "mean"). The exact nature and shape of the output tensors depend on the underlying aggregation implementation.
 *
 * @note This function resides in the `pyc::graph::cartesian` namespace.
 * @note Internally, the separate Cartesian components (`px`, `py`, `pz`, `e`) are likely combined into a feature tensor before aggregation.
 * @note Input tensors are managed for device placement via `changedev`.
 * @note The core aggregation logic is handled by `graph_::edge_aggregation`.
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::edge_aggregation(
    torch::Tensor edge_index, torch::Tensor prediction,
    torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
);

/**
 * @brief Performs node aggregation for a graph using explicitly provided Cartesian coordinate components.
 * @details This function aggregates features at the node level. It takes the graph structure (`edge_index`),
 * edge-level predictions or scores (`prediction`), and node kinematic variables provided as separate Cartesian
 * components (`px`, `py`, `pz`, `e`). These Cartesian components serve as the base node features. The function
 * uses an underlying node aggregation mechanism (`graph_::node_aggregation`) that aggregates features from neighboring
 * nodes based on the graph connectivity, potentially weighting contributions using the `prediction` scores.
 * The results of different aggregation types (e.g., sum, mean) are returned in a dictionary.
 *
 * @param edge_index A tensor of shape [2, num_edges] representing the edge indices of the graph (source and target nodes).
 *                   Defines neighborhood relationships. Must be on the correct device.
 * @param prediction A tensor containing prediction scores or weights, likely associated with edges, used to modulate aggregation.
 *                   Shape typically [num_edges] or [num_edges, num_prediction_features]. Must be on the correct device.
 * @param px A 1D tensor of length num_nodes representing the x-component of momentum for each node. Must be on the correct device.
 * @param py A 1D tensor of length num_nodes representing the y-component of momentum for each node. Must be on the correct device.
 * @param pz A 1D tensor of length num_nodes representing the z-component of momentum for each node. Must be on the correct device.
 * @param e A 1D tensor of length num_nodes representing the energy of each node. Must be on the correct device.
 *
 * @return A torch::Dict<std::string, torch::Tensor> containing the aggregated node features, keyed by aggregation type
 *         (e.g., "sum", "mean"). Output tensors typically have shape [num_nodes, aggregated_feature_size].
 *
 * @note This function resides in the `pyc::graph::cartesian` namespace.
 * @note Internally, the separate Cartesian components (`px`, `py`, `pz`, `e`) are combined into a node feature tensor before aggregation.
 * @note Input tensors are managed for device placement via `changedev`.
 * @note The core aggregation logic is handled by `graph_::node_aggregation`.
 */
torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::node_aggregation(
    torch::Tensor edge_index, torch::Tensor prediction,
    torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
);

