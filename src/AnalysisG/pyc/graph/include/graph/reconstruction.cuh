/**
 * @file reconstruction.cuh
 * @brief Provides CUDA kernel definitions for graph reconstruction operations.
 */

#ifndef GRAPH_RECONSTRUCTION_CUH
#define GRAPH_RECONSTRUCTION_CUH

#include <torch/torch.h> ///< Includes PyTorch C++ headers for tensor operations.
#include <utils/utils.cuh> ///< Includes utility functions for CUDA operations.

/**
 * @brief Namespace for graph-related operations.
 */
namespace graph_ {

/**
 * @brief CUDA kernel for reconstructing graph edges.
 *
 * @tparam scalar_t The data type of the tensor elements (e.g., float, double).
 * @param edge_index Input tensor containing edge indices of the graph.
 * @param features Input tensor containing node features.
 * @param reconstructed Output tensor for reconstructed edges.
 * @param num_nodes The number of nodes in the graph.
 */
template <typename scalar_t>
__global__ void reconstruct_edges_kernel(
    const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_index,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> features,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> reconstructed,
    const unsigned int num_nodes
);

/**
 * @brief Reconstructs the edges of a graph based on node features.
 *
 * @param edge_index Input tensor containing edge indices of the graph.
 * @param features Input tensor containing node features.
 * @return A tensor containing the reconstructed edges.
 */
torch::Tensor reconstruct_edges(torch::Tensor edge_index, torch::Tensor features);

} // namespace graph_

#endif // GRAPH_RECONSTRUCTION_CUH
