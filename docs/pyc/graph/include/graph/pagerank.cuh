/**
 * @file pagerank.cuh
 * @brief Defines functions for calculating PageRank scores on graphs using CUDA and PyTorch tensors.
 *
 * This header provides interfaces for standard PageRank computation as well as a variant
 * incorporating a reconstruction component. The calculations are designed to be performed
 * efficiently on GPU hardware.
 */

/**
 * @namespace graph_
 * @brief Encapsulates graph-related algorithms and utilities.
 */
namespace graph_ {
        /**
         * @brief Computes the PageRank scores for each node in a given graph.
         * @details This function implements the iterative PageRank algorithm. It calculates the
         * importance score for each node based on the structure of the graph represented by
         * `edge_index` and optional edge weights `edge_scores`. The algorithm iterates until
         * the change in scores between iterations falls below the specified `threshold` or
         * the maximum number of iterations (`timeout`) is reached. The scores are influenced
         * by the damping factor `alpha`. The resulting PageRank scores can be optionally
         * normalized using `norm_low`.
         *
         * The PageRank formula is generally defined as:
         * PR(u) = (1 - alpha) / N + alpha * sum(PR(v) / L(v))
         * where PR(u) is the PageRank of node u, N is the total number of nodes,
         * alpha is the damping factor, v are the nodes linking to u, and L(v) is the
         * number of outbound links from node v (or sum of weights if weighted).
         * This implementation uses PyTorch tensors for input and output, enabling
         * integration with PyTorch-based workflows and leveraging GPU acceleration via CUDA.
         *
         * @param[in] edge_index A pointer to a `torch::Tensor` of shape [2, num_edges] and type `torch::kLong`.
         *                       It represents the graph's connectivity in COO (Coordinate List) format,
         *                       where `edge_index[0]` contains source nodes and `edge_index[1]` contains
         *                       destination nodes for each edge. The graph is assumed to be directed.
         * @param[in] edge_scores A pointer to a `torch::Tensor` of shape [num_edges] and a floating-point type
         *                        (e.g., `torch::kFloat` or `torch::kDouble`). It represents the weight or score
         *                        associated with each edge in `edge_index`. If edge weights are not used,
         *                        this can be set appropriately (e.g., a tensor of ones or nullptr, depending
         *                        on implementation details - check function source).
         * @param[in] alpha The damping factor (teleportation probability), typically between 0.8 and 0.9.
         *                  It represents the probability that a random surfer will continue clicking links
         *                  versus jumping to a random page. A value of 1.0 means no random jumps, while 0.0
         *                  means only random jumps.
         * @param[in] threshold The convergence threshold. The iterative algorithm stops when the maximum
         *                      absolute difference between PageRank scores in successive iterations is less
         *                      than this value. A smaller value leads to higher accuracy but potentially
         *                      more iterations.
         * @param[in] norm_low The lower bound for normalization applied to the final PageRank scores.
         *                     If normalization is applied, scores might be scaled to a specific range
         *                     (e.g., [norm_low, 1.0]). Check implementation for exact normalization method.
         * @param[in] timeout The maximum number of iterations allowed for the PageRank algorithm.
         *                    This prevents infinite loops in case the algorithm does not converge
         *                    within a reasonable number of steps.
         * @param[in] num_cls The number of classes or partitions. The exact role of this parameter
         *                    depends on the specific implementation details. It might be used for
         *                    personalized PageRank variants or graph partitioning schemes related
         *                    to the PageRank calculation.
         *
         * @return A `std::map<std::string, torch::Tensor>`. The map typically contains one entry
         *         where the key is a descriptive string (e.g., "pagerank") and the value is a
         *         `torch::Tensor` of shape [num_nodes] containing the calculated PageRank score
         *         for each node in the graph. The data type of the tensor will likely be a
         *         floating-point type.
         *
         * @note The input tensors (`edge_index`, `edge_scores`) are expected to reside on the
         *       appropriate device (CPU or CUDA GPU) where the computation is intended to run.
         * @warning Ensure that `edge_index` correctly represents the graph structure and that
         *          `edge_scores` (if provided) has the correct dimensions corresponding to the edges.
         *          Invalid inputs can lead to incorrect results or runtime errors.
         */
        std::map<std::string, torch::Tensor> page_rank(
                        torch::Tensor* edge_index, torch::Tensor* edge_scores,
                        double alpha, double threshold, double norm_low, long timeout, int num_cls
        );

        /**
         * @brief Computes PageRank scores incorporating a reconstruction component.
         * @details This function extends the standard PageRank calculation by integrating a
         * reconstruction term, represented by the `pmc` tensor. This variant might be used
         * in scenarios where PageRank scores need to be influenced by or aligned with some
         * prior information, target distribution, or reconstruction objective related to the nodes.
         * The exact mathematical formulation combining PageRank with reconstruction depends
         * on the specific implementation. Like the standard `page_rank` function, it uses an
         * iterative approach controlled by `alpha`, `threshold`, and `timeout`.
         *
         * @param[in] edge_index A pointer to a `torch::Tensor` of shape [2, num_edges] and type `torch::kLong`.
         *                       Represents the graph connectivity in COO format. (See `page_rank` for details).
         * @param[in] edge_scores A pointer to a `torch::Tensor` of shape [num_edges] and a floating-point type.
         *                        Represents the weight associated with each edge. (See `page_rank` for details).
         * @param[in] pmc A pointer to a `torch::Tensor`. The shape and content of this tensor depend
         *                on the specific reconstruction mechanism being implemented. It might represent
         *                personalization vectors, prior node scores, or target values used to modify
         *                the PageRank update step or convergence criteria. Its data type should be
         *                compatible with the PageRank score calculations (likely float or double).
         * @param[in] alpha The damping factor for the PageRank component of the calculation.
         *                  (See `page_rank` for details).
         * @param[in] threshold The convergence threshold for the iterative calculation. The stopping
         *                      criterion might involve both the change in PageRank scores and the
         *                      reconstruction error, depending on the implementation.
         * @param[in] norm_low The lower bound for normalization applied to the final scores.
         *                     (See `page_rank` for details).
         * @param[in] timeout The maximum number of iterations allowed for the algorithm.
         *                    (See `page_rank` for details).
         * @param[in] num_cls The number of classes or partitions. Similar to `page_rank`, its role
         *                    depends on the specific implementation, potentially influencing the
         *                    reconstruction process or personalization aspects.
         *
         * @return A `std::map<std::string, torch::Tensor>`. Similar to `page_rank`, this map
         *         contains the resulting scores as a `torch::Tensor` associated with a descriptive key.
         *         The scores reflect the PageRank influenced by the reconstruction term `pmc`.
         *
         * @note The input tensors (`edge_index`, `edge_scores`, `pmc`) must be on the correct device.
         * @warning The interpretation of the results depends heavily on the specific reconstruction
         *          method implemented within the function. Ensure the `pmc` tensor is correctly
         *          formatted and represents the intended reconstruction information.
         * @see page_rank() for the standard PageRank calculation without reconstruction.
         */
        std::map<std::string, torch::Tensor> page_rank_reconstruction(
                        torch::Tensor* edge_index, torch::Tensor* edge_scores, torch::Tensor* pmc,
                        double alpha, double threshold, double norm_low, long timeout, int num_cls
        );


} // namespace graph_

#endif
