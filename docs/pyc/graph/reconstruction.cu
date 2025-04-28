/**
 * @file reconstruction.cu
 * @brief Implements CUDA kernels and host functions for reconstructing graph structures based on PageRank scores.
 *
 * This file contains the core logic for identifying unique nodes within a graph representation and performing
 * a PageRank-based reconstruction. It leverages CUDA for parallel computation to efficiently process large graphs.
 * The primary goal is to cluster nodes and aggregate their properties based on PageRank values.
 */

#include <torch/extension.h> // PyTorch C++ extension API for tensor manipulation and CUDA integration.
#include <map>              // Standard C++ map container, used for returning multiple named tensors.
#include <string>           // Standard C++ string class, used for keys in the return map.

/**
 * @brief CUDA kernel to identify unique nodes and aggregate their associated properties.
 *
 * This kernel processes a list of nodes, potentially with duplicates, and identifies the unique nodes.
 * For each unique node, it aggregates associated data like particle momentum components (pmc),
 * PageRank cluster values (pgrc), PageRank node values (pgrk), and tracking indices (trk).
 * It utilizes shared memory for efficient parallel reduction and aggregation within thread blocks.
 * The results, including aggregated PageRank values and momentum components for unique nodes,
 * are written to output tensors.
 *
 * @tparam scalar_t The data type used for PageRank scores and potentially other floating-point calculations (e.g., float, double).
 *                  This allows flexibility in precision requirements.
 * @tparam size_x   A compile-time constant specifying the size of the shared memory arrays used within the kernel.
 *                  This typically corresponds to the block size or a related dimension optimized for the target GPU architecture.
 *
 * @param node_index A 1D tensor accessor providing read access to the indices of nodes in the graph.
 *                   Each element represents a node identifier, which may appear multiple times.
 * @param count A 3D tensor accessor providing read/write access to counts associated with nodes, potentially across different dimensions or categories.
 *              Its exact usage depends on the broader algorithm context but likely involves tracking occurrences or properties.
 * @param pmc A 2D tensor accessor providing read access to particle momentum components (e.g., px, py, pz, E) associated with each node entry in `node_index`.
 * @param pgrc A 2D tensor accessor providing read access to PageRank cluster scores associated with each node entry.
 * @param pgrk A 2D tensor accessor providing read access to PageRank node scores associated with each node entry.
 * @param trk A 2D tensor accessor providing read access to tracking indices or identifiers associated with each node entry.
 * @param page_cluster A 2D tensor accessor for writing the aggregated PageRank cluster scores for each unique node identified by the kernel.
 * @param page_node A 2D tensor accessor for writing the aggregated PageRank node scores for each unique node identified by the kernel.
 * @param pmc_out A 3D tensor accessor for writing the aggregated particle momentum components for each unique node identified by the kernel.
 * @param num_n The total number of node entries in the `node_index` tensor (i.e., the size of the first dimension). This determines the total workload.
 */
template <typename scalar_t, size_t size_x>
__global__ void _find_unique(
    torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> node_index,
    torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> count,
    torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> pmc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pgrc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pgrk,
    torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> trk,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> page_cluster,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> page_node,
    torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> pmc_out,
    const unsigned long num_n
);

/**
 * @brief Host function to orchestrate the PageRank-based graph reconstruction process.
 *
 * This function serves as the main entry point for the graph reconstruction algorithm.
 * It takes graph connectivity information (edge index and scores), node properties (particle momentum),
 * and PageRank algorithm parameters as input. It likely involves:
 * 1. Preprocessing the input data.
 * 2. Potentially running a PageRank algorithm (or utilizing precomputed scores passed via `pgrc`, `pgrk` - the exact implementation detail is not shown here but implied by the parameters of `_find_unique`).
 * 3. Launching the `_find_unique` CUDA kernel to identify unique nodes and aggregate their properties based on PageRank scores.
 * 4. Post-processing the kernel results.
 * 5. Returning the reconstructed graph information, typically including aggregated node properties and potentially cluster assignments, packaged in a map of tensors.
 *
 * @param edge_index A pointer to a tensor representing the graph's edge connectivity. Typically a 2xN tensor where N is the number of edges,
 *                   with the first row containing source node indices and the second row containing target node indices.
 * @param edge_scores A pointer to a tensor containing scores or weights associated with each edge in `edge_index`.
 *                    These scores might influence the PageRank calculation or subsequent aggregation steps.
 * @param pmc A pointer to a tensor containing particle momentum components associated with the nodes in the graph.
 *            This data is aggregated by the `_find_unique` kernel.
 * @param alpha The damping factor (teleportation probability) used in the PageRank algorithm. Typically a value between 0.8 and 0.9.
 *              Controls the balance between following graph links and randomly jumping to any node.
 * @param threshold The convergence threshold for the iterative PageRank calculation. The algorithm stops when the change in PageRank scores between iterations falls below this value.
 * @param norm_low A normalization parameter, potentially used to scale PageRank scores or other intermediate values to a specific range (e.g., [norm_low, 1]).
 * @param timeout A maximum time limit or iteration count for the PageRank computation to prevent indefinite execution if convergence is slow or fails.
 * @param num_cls The desired or expected number of clusters to be identified or used during the reconstruction process. This might influence how PageRank scores are interpreted or aggregated.
 *
 * @return A `std::map<std::string, torch::Tensor>` where keys are descriptive strings (e.g., "unique_pmc", "unique_pagerank_cluster")
 *         and values are PyTorch tensors containing the results of the reconstruction process, such as the aggregated properties of the unique nodes.
 */
std::map<std::string, torch::Tensor> graph_::page_rank_reconstruction(
    torch::Tensor* edge_index,
    torch::Tensor* edge_scores,
    torch::Tensor* pmc,
    double alpha,
    double threshold,
    double norm_low,
    long timeout,
    int num_cls
);

