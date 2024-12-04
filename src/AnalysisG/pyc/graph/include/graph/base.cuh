#ifndef CUGRAPH_BASE_H
#define CUGRAPH_BASE_H
#include <atomic/cuatomic.cuh>

template <typename scalar_t>
__global__ void _prediction_topology(
              torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> pairs, 
        const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_index, 
        const torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> pred,
        const unsigned int dx_lx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= dx_lx){return;}
    long prd = pred[idx]; 
    long src = edge_index[0][idx]; 
    long dst = edge_index[1][idx]; 
    pairs[prd][src][dst] = dst; 
    if (src != dst){return;}
    for (size_t x(0); x < pairs.size({0}); ++x){pairs[x][src][dst] = src;}

}

template <typename scalar_t>
__global__ void _edge_summing(
              torch::PackedTensorAccessor64<long    , 3, torch::RestrictPtrTraits> pairs, 
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> pmu, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmi, 
        const unsigned int pred_lx, const unsigned int node_lx, const unsigned int node_fx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z; 
    if (idx >= node_lx || idy >= node_fx || idz >= pred_lx){return;}

    double sx = 0; 
    for (size_t x(0); x < node_lx; ++x){sx += (pairs[idz][idx][x] >= 0) * pmi[x][idy];}
    pmu[idz][idx][idy] = sx; 
}


template <typename scalar_t> 
__global__ void _fast_unique(
              torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> out_map, 
        const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> cluster_map, 
        const unsigned int dim_i, const unsigned int dim_j, const unsigned int dim_k
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z; 
    if (idx >= dim_i || idy >= dim_j || idz >= dim_k || idz >= idy){return;}   
    if (cluster_map[idx][idy] < 0){return;}
    if (cluster_map[idx][idz] < 0){return;}
    if (!(cluster_map[idx][idy] == cluster_map[idx][idz])){return;}
    out_map[idx][idy] = -1; 
}


template <typename scalar_t>
__global__ void _unique_sum(
              torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<long    , 2, torch::RestrictPtrTraits> cluster_map, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> features, 
        const unsigned int dim_i, const unsigned int dim_j, const unsigned int dim_k
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; 
    if (idx >= dim_i || idy >= dim_k){return;}

    scalar_t sx = 0; 
    for (unsigned int i(0); i < dim_j; ++i){
        const long tx = cluster_map[idx][i]; 
        if (tx < 0){continue;}
        sx += features[ tx ][idy];   
    }
    out[idx][idy] = sx; 
}



#endif
