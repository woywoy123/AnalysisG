/**
 * @file base.cuh
 * @brief Contains CUDA kernel definitions for graph operations in the AnalysisG framework.
 *
 * This file defines CUDA kernels for processing graph prediction data, aggregating edge features,
 * identifying unique elements in a cluster map, and computing the sum of features for unique clusters.
 */

#ifndef CUGRAPH_BASE_H
#define CUGRAPH_BASE_H

#include <utils/atomic.cuh> ///< Includes atomic utilities for CUDA operations.

/**
 * @brief CUDA kernel for processing graph prediction data.
 *
 * @tparam scalar_t The data type of the tensor elements (e.g., float, double).
 * @param pairs Output tensor for pairs, 3D tensor of longs.
 * @param edge_index Input tensor for edge indices, 2D tensor of longs.
 * @param pred Input tensor for predictions, 1D tensor of longs.
 * @param dx_lx Input unsigned integer, likely size or dimension.
 */
template <typename scalar_t>
__global__ void _prediction_topology(
    torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> pairs,
    const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_index,
    const torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> pred,
    const unsigned int dx_lx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; ///< Calculate unique thread index.
    if (idx >= dx_lx) { return; } ///< Boundary check: if index is out of bounds, return.
    long prd = pred[idx]; ///< Get prediction value at current index.
    long src = edge_index[0][idx]; ///< Get source node from edge_index.
    long dst = edge_index[1][idx]; ///< Get destination node from edge_index.
    pairs[prd][src][dst] = dst; ///< Assign destination to the pairs tensor based on prediction, source, and destination.
    if (src != dst) { return; } ///< If source and destination are different, return.
    for (size_t x(0); x < pairs.size({0}); ++x) { pairs[x][src][dst] = src; } ///< Assign source to pairs tensor for all predictions at this src/dst.
}

/**
 * @brief CUDA kernel for aggregating edge features.
 *
 * @tparam scalar_t The data type of the tensor elements.
 * @param pairs Input tensor for pairs, 3D tensor of longs.
 * @param pmu Output tensor for aggregated results, 3D tensor of scalar_t.
 * @param pmi Input tensor for features, 2D tensor of scalar_t.
 * @param pred_lx Input unsigned integer, likely number of predictions.
 * @param node_lx Input unsigned integer, likely number of nodes.
 * @param node_fx Input unsigned integer, likely number of features.
 */
template <typename scalar_t>
__global__ void _edge_summing(
    torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> pairs,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> pmu,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmi,
    const unsigned int pred_lx,
    const unsigned int node_lx,
    const unsigned int node_fx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; ///< Calculate unique thread index for nodes.
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; ///< Calculate unique thread index for features.
    const unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z; ///< Calculate unique thread index for predictions.
    if (idx >= node_lx || idy >= node_fx || idz >= pred_lx) { return; } ///< Boundary check: if indices are out of bounds, return.

    double sx = 0; ///< Initialize sum variable.
    for (size_t x(0); x < node_lx; ++x) { sx += (pairs[idz][idx][x] >= 0) * pmi[x][idy]; } ///< Sum over valid pairs and features.
    pmu[idz][idx][idy] = sx; ///< Assign sum to output tensor.
}

/**
 * @brief CUDA kernel for identifying unique elements in a cluster map.
 *
 * @tparam scalar_t The data type of the tensor elements.
 * @param out_map Output tensor for unique map, 2D tensor of longs.
 * @param cluster_map Input tensor for cluster mapping, 2D tensor of longs.
 * @param dim_i Input unsigned integer, likely dimension i.
 * @param dim_j Input unsigned integer, likely dimension j.
 * @param dim_k Input unsigned integer, likely dimension k.
 */
template <typename scalar_t>
__global__ void _fast_unique(
    torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> out_map,
    const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> cluster_map,
    const unsigned int dim_i,
    const unsigned int dim_j,
    const unsigned int dim_k
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; ///< Calculate unique thread index for dimension i.
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; ///< Calculate unique thread index for dimension j.
    const unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z; ///< Calculate unique thread index for dimension k.
    if (idx >= dim_i || idy >= dim_j || idz >= dim_k || idz >= idy) { return; } ///< Boundary check: if indices are out of bounds, return.
    if (cluster_map[idx][idy] < 0) { return; } ///< Skip if cluster_map value is negative.
    if (cluster_map[idx][idz] < 0) { return; } ///< Skip if cluster_map value is negative.
    if (!(cluster_map[idx][idy] == cluster_map[idx][idz])) { return; } ///< Skip if values are not equal.
    out_map[idx][idy] = -1; ///< Mark as non-unique.
}

/**
 * @brief CUDA kernel for computing the sum of features for unique clusters.
 *
 * @tparam scalar_t The data type of the tensor elements.
 * @param out Output tensor for results, 2D tensor of scalar_t.
 * @param cluster_map Input tensor for cluster mapping, 2D tensor of longs.
 * @param features Input tensor for features, 2D tensor of scalar_t.
 * @param dim_i Input unsigned integer, likely dimension i.
 * @param dim_j Input unsigned integer, likely dimension j.
 * @param dim_k Input unsigned integer, likely dimension k.
 */
template <typename scalar_t>
__global__ void _unique_sum(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out,
    const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> cluster_map,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> features,
    const unsigned int dim_i,
    const unsigned int dim_j,
    const unsigned int dim_k
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; ///< Calculate unique thread index for dimension i.
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; ///< Calculate unique thread index for dimension k.
    if (idx >= dim_i || idy >= dim_k) { return; } ///< Boundary check: if indices are out of bounds, return.

    scalar_t sx = 0; ///< Initialize sum variable.
    for (unsigned int i(0); i < dim_j; ++i) { ///< Loop through dimension j.
        const long tx = cluster_map[idx][i]; ///< Get value from cluster_map.
        if (tx < 0) { continue; } ///< Skip if value is negative.
        sx += features[tx][idy]; ///< Add feature value to sum.
    }
    out[idx][idy] = sx; ///< Assign sum to output tensor.
}

#endif // CUGRAPH_BASE_H
