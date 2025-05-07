/**
 * @file pagerank.cu
 * @brief This file provides the CUDA kernel implementations for the PageRank algorithm
 *        applied to graph analysis. It includes kernels for preprocessing steps like
 *        finding the maximum node ID and remapping indices, as well as the main
 *        PageRank computation kernel.
 */

#include <graph/pagerank.cuh> // Include header file for PageRank declarations
#include <utils/atomic.cuh>   // Include utilities for atomic operations on GPU
#include <utils/utils.cuh>    // Include general utility functions

/**
 * @brief CUDA kernel to determine the maximum node ID present in the edge list
 *        and create a mapping for potential re-indexing.
 *
 * @details This kernel processes the input edge list (`edge_inx`) in parallel.
 *          Each thread block cooperatively finds the maximum node ID within its
 *          assigned portion of the edges using shared memory for efficiency.
 *          The maximum node ID found by each block is stored in the `max_node` tensor.
 *          Optionally, it can also populate a `remap` tensor, which might be used
 *          in subsequent steps for compacting the node ID space if needed.
 *
 * @tparam size_x Specifies the number of threads per block along the x-dimension.
 *                This parameter influences the amount of shared memory used and
 *                the parallelism strategy within the kernel.
 *
 * @param edge_inx A packed tensor accessor representing the input graph edges.
 *                 It's expected to have a shape of [2, num_edges], where `edge_inx[0]`
 *                 contains the source nodes and `edge_inx[1]` contains the destination
 *                 nodes for each edge. The data type is `long`.
 * @param max_node A packed tensor accessor for the output tensor where each block
 *                 writes the maximum node ID it encountered. This is typically used
 *                 in a reduction step afterwards to find the global maximum.
 *                 The data type is `long`.
 * @param remap A packed tensor accessor for an output tensor intended to store
 *              remapping indices. The exact usage depends on the overall algorithm design.
 *              The data type is `long`.
 * @param el The total number of edges in the graph (`num_edges`). This determines
 *           the total workload for the kernel.
 * @param mxn An initial estimate or upper bound for the maximum node ID. This might
 *            be used for initialization or bounds checking within the kernel.
 */
template <size_t size_x>
__global__ void _get_max_node(
    const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_inx,
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> max_node,
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> remap,
          const int el, const long mxn);

/**
 * @brief CUDA kernel to create a remapping of node indices based on evaluation nodes.
 *
 * @details This kernel is responsible for creating a mapping between the original
 *          node IDs in the graph (`edge_inx`) and a potentially new, possibly
 *          compacted, index space. It considers a specific set of nodes to be
 *          evaluated (`ev_node`). The mapping is stored in `idx_map`. It also
 *          computes batch information (`num_batch`) and node counts (`num_enode`)
 *          related to the evaluation nodes and the remapping process. This is
 *          often used to prepare data structures for efficient processing of
 *          specific subsets of the graph or for batch processing.
 *
 * @param edge_inx A packed tensor accessor for the input edge list [2, num_edges].
 *                 Contains the source and destination nodes of the graph. Data type `long`.
 * @param ev_node A packed tensor accessor containing the list of node IDs that are
 *                specifically targeted for evaluation or processing. Data type `long`.
 * @param idx_map A packed tensor accessor for the input/output index mapping. It will
 *                be populated by this kernel to map original node IDs to new indices.
 *                Data type `long`.
 * @param num_batch A packed tensor accessor for an output tensor storing information
 *                  about batches, likely related to how `ev_node`s are grouped.
 *                  Data type `long`.
 * @param num_enode A packed tensor accessor for an output tensor storing counts related
 *                  to the evaluation nodes after remapping. Data type `long`.
 * @param mxn The maximum node ID found in the original graph (`edge_inx`). Used for
 *            determining the size of mapping arrays or loops.
 * @param mxn_ev The maximum node ID present in the `ev_node` list.
 */
__global__ void _get_remapping(
    const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_inx,
    const torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> ev_node,
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> idx_map,
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> num_batch,
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> num_enode,
          const long mxn, const long mxn_ev);

/**
 * @brief CUDA kernel to compute the PageRank scores for nodes in a graph and
 *        optionally perform clustering based on these scores.
 *
 * @details This is the core kernel for the PageRank computation. It iteratively
 *          updates the PageRank score for each node based on the scores of its
 *          neighbors and the edge weights (`edge_scores`), incorporating the damping
 *          factor (`alpha`). The computation is often performed in batches or focused
 *          on specific events defined by `cu_xms` and `cu_xme`. Convergence is
 *          determined by the `threshold` parameter or a maximum number of iterations
 *          (`timeout`).
 *          The kernel can also identify clusters of nodes based on PageRank scores
 *          and connectivity, storing cluster-level PageRank (`pageclus`), node counts
 *          per cluster (`count`), and marking edges belonging to identified clusters
 *          (`edge_inx`). The `mlp_lim` and `min_nodes` parameters control aspects
 *          of the clustering process. Shared memory (`size_x`) is used for efficient
 *          data sharing within thread blocks during the iterative updates.
 *
 * @tparam scalar_t The data type used for PageRank scores and edge weights (e.g., `float`, `double`).
 * @tparam size_x Specifies the number of threads per block along the x-dimension,
 *                influencing shared memory usage and parallel execution strategy.
 *
 * @param cu_xms A packed tensor accessor containing the starting indices or offsets
 *               for processing different events or batches within the graph data. Data type `long`.
 * @param cu_xme A packed tensor accessor containing node counts or ending indices
 *               corresponding to the events/batches defined by `cu_xms`. Data type `long`.
 * @param edge_scores A packed tensor accessor holding the scores or weights associated
 *                    with each edge. Shape [2, num_edges] or similar, depending on
 *                    whether scores are directional. Data type `scalar_t`.
 * @param pagerank A packed tensor accessor for the output tensor storing the computed
 *                 PageRank score for each node. Shape [num_events/batches, max_node_id].
 *                 Data type `scalar_t`.
 * @param pageclus A packed tensor accessor for the output tensor storing aggregated
 *                 PageRank scores at the cluster level. Shape might depend on the
 *                 number of clusters found. Data type `scalar_t`.
 * @param count A packed tensor accessor for a 3D output tensor storing node counts
 *              within identified clusters, potentially per event/batch. Data type `long`.
 * @param edge_inx A packed tensor accessor for a boolean output tensor indicating whether
 *                 an edge is part of a detected cluster. Shape [num_edges]. Data type `bool`.
 * @param alpha The damping factor for the PageRank algorithm, typically around 0.85.
 *              Represents the probability of following a link versus jumping to a random node.
 * @param mlp_lim A minimum likelihood probability limit, potentially used in the
 *                clustering part of the algorithm to filter connections or nodes.
 * @param threshold The convergence threshold. The iterative process stops when the
 *                  maximum change in PageRank scores between iterations falls below this value.
 * @param mx_ev The maximum number of evaluation nodes considered, possibly related to `ev_node`
 *              in `_get_remapping` or the size of batches.
 * @param num_ev The number of distinct events or batches to process, corresponding to
 *               the entries in `cu_xms` and `cu_xme`.
 * @param min_nodes The minimum number of nodes required to form a valid cluster during
 *                  the clustering phase.
 * @param mxn The maximum node ID in the graph (or the remapped space). Used for sizing
 *            arrays and loops.
 * @param timeout The maximum number of iterations allowed for the PageRank computation
 *                to converge. Prevents infinite loops if convergence is not reached.
 */
template <typename scalar_t, size_t size_x>
__global__ void _page_rank(
    const torch::PackedTensorAccessor64<long    , 1, torch::RestrictPtrTraits> cu_xms,
    const torch::PackedTensorAccessor64<long    , 1, torch::RestrictPtrTraits> cu_xme,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> edge_scores,
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pagerank,
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pageclus,
          torch::PackedTensorAccessor64<long    , 3, torch::RestrictPtrTraits> count,
          torch::PackedTensorAccessor64<bool    , 1, torch::RestrictPtrTraits> edge_inx,
    const double alpha,
    const double mlp_lim,
    const double threshold,
    const unsigned int mx_ev,
    const unsigned int num_ev,
    const unsigned int min_nodes,
    const unsigned int mxn,
    const long timeout);

/**
 * @brief Host function to orchestrate the PageRank computation on the GPU.
 *
 * @details This function serves as the main entry point for performing the PageRank
 *          analysis. It takes PyTorch tensors containing the graph structure
 *          (`edge_index`) and edge weights (`edge_scores`) as input. It manages
 *          the allocation of GPU memory, potentially calls preprocessing kernels
 *          like `_get_max_node` or `_get_remapping` (though not explicitly shown
 *          in the declaration, it's implied by the overall workflow), configures
 *          and launches the main `_page_rank` kernel, and handles data transfer
 *          between the host (CPU) and the device (GPU). Finally, it gathers the
 *          results (PageRank scores, cluster information, edge masks) into
 *          PyTorch tensors and returns them in a map.
 *
 * @param edge_index A pointer to a PyTorch tensor representing the graph's edge list.
 *                   Expected shape [2, num_edges], containing source and destination
 *                   node indices. Data type typically `torch::kLong`.
 * @param edge_scores A pointer to a PyTorch tensor containing the scores or weights
 *                    for each edge. Shape should be compatible with `edge_index`.
 *                    Data type typically `torch::kFloat` or `torch::kDouble`.
 * @param alpha The damping factor for the PageRank algorithm (e.g., 0.85).
 * @param threshold The convergence threshold for the iterative PageRank calculation.
 * @param norm_low A parameter likely related to normalization or filtering, possibly
 *                 equivalent to `mlp_lim` used in the kernel.
 * @param timeout The maximum number of iterations allowed for convergence.
 * @param num_cls The minimum number of nodes required to form a valid cluster,
 *                corresponding to `min_nodes` in the kernel.
 *
 * @return std::map<std::string, torch::Tensor> A map where keys are strings
 *         describing the output tensors and values are the corresponding PyTorch
 *         tensors computed on the GPU. Expected keys include:
 *         - "nodes": Tensor possibly containing node counts per cluster or similar cluster stats.
 *         - "pagerank": Tensor containing the final PageRank score for each node.
 *         - "pagenode": Tensor likely containing cluster-level PageRank aggregates (`pageclus`).
 *         - "edge_mask": Boolean tensor indicating which edges belong to identified clusters.
 *         - "node_index": Tensor possibly related to batching or event indexing (`cu_xms`).
 */
std::map<std::string, torch::Tensor> graph_::page_rank(
    torch::Tensor* edge_index, torch::Tensor* edge_scores,
    double alpha, double threshold, double norm_low, long timeout, int num_cls);

