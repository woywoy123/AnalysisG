/**
 * @file pagerank.cuh
 * @brief Provides CUDA kernel definitions for the PageRank algorithm.
 */

#ifndef GRAPH_PAGERANK_CUH
#define GRAPH_PAGERANK_CUH

#include <torch/torch.h> ///< Includes PyTorch C++ headers for tensor operations.
#include <utils/utils.cuh> ///< Includes utility functions for CUDA operations.

/**
 * @brief Namespace for graph-related operations.
 */
namespace graph_ {

/**
 * @brief CUDA kernel for computing PageRank scores.
 *
 * @tparam scalar_t The data type of the tensor elements (e.g., float, double).
 * @param edge_index Input tensor containing edge indices of the graph.
 * @param rank Input tensor containing initial rank values.
 * @param out Output tensor for updated rank values.
 * @param alpha Damping factor for the PageRank algorithm.
 * @param num_nodes The number of nodes in the graph.
 */
template <typename scalar_t>
__global__ void pagerank_kernel(
    const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_index,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> rank,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> out,
    const scalar_t alpha,
    const unsigned int num_nodes
);

/**
 * @brief Computes the PageRank scores for a graph.
 *
 * @param edge_index Input tensor containing edge indices of the graph.
 * @param alpha Damping factor for the PageRank algorithm.
 * @param num_iterations The number of iterations to run the algorithm.
 * @return A tensor containing the PageRank scores for each node.
 */
torch::Tensor pagerank(torch::Tensor edge_index, double alpha, int num_iterations);

} // namespace graph_

#endif // GRAPH_PAGERANK_CUH
