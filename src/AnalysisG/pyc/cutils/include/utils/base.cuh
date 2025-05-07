/**
 * @file base.cuh
 * @brief Provides base definitions for CUDA utilities in the cutils module.
 */

#ifndef CUTILS_BASE_CUH
#define CUTILS_BASE_CUH

#include <torch/torch.h> ///< Includes PyTorch C++ headers for tensor operations.

/**
 * @brief Namespace for CUDA utility functions.
 */
namespace cutils {

/**
 * @brief Computes the number of blocks required for a given number of threads.
 *
 * @param total_threads The total number of threads.
 * @param threads_per_block The number of threads per block.
 * @return The number of blocks required.
 */
__host__ __device__ inline int compute_blocks(int total_threads, int threads_per_block) {
    return (total_threads + threads_per_block - 1) / threads_per_block;
}

} // namespace cutils

#endif // CUTILS_BASE_CUH
