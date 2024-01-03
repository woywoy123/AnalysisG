#include <torch/torch.h>

template <typename scalar_t>
__global__ void _PredTopo(
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> edge_index, 
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> pred, 
        const unsigned int dim_i, const unsigned int dim_max, const bool incl_z)
{
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    if (idx >= dim_i || idy >= dim_max){ return; }

    const unsigned int src = edge_index[0][idx]; 
    const unsigned int dst = edge_index[1][idx]; 
    scalar_t* val = &out[idy][src][dst]; 

    if (!incl_z && idy == 0){return; }
    if (src == dst){*val = src; return; }
    
    scalar_t msk = (pred[idx] == idy); 
    if (!msk){ return; }
    *val = dst;  
}

template <typename scalar_t>
__global__ void _EdgeSummation(
        torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> pmu_i, 
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> triggers, 
        const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> pmu, 
        const unsigned int dim_i, const unsigned int dim_j, const unsigned int dim_max)
{
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_i || idy >= dim_j || idz >= dim_max){ return; }
   
    for (int i(0); i < dim_i; ++i){
        if (triggers[idz][idx][i] < 0){ continue; }
        pmu_i[idz][idx][idy] += pmu[ triggers[idz][idx][i] ][idy];
    }
}

template <typename scalar_t> 
__global__ void _fast_unique(
        torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> out_map, 
        const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> cluster_map, 
        const unsigned int dim_i, const unsigned int dim_j, const unsigned int dim_k)
{
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_i || idy >= dim_j || idz >= dim_k || idz >= idy){return;}   
    if (cluster_map[idx][idy] < 0){return;}
    if (cluster_map[idx][idz] < 0){return;}

    bool msk = cluster_map[idx][idy] == cluster_map[idx][idz];
    if (!msk){return;}
    out_map[idx][idy] = -1; 
}


template <typename scalar_t>
__global__ void _unique_sum(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> cluster_map, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> features, 
        const unsigned int dim_i, const unsigned int dim_j, const unsigned int dim_k)
{

    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    if (idx >= dim_i || idy >= dim_k){return;}

    for (unsigned int i(0); i < dim_j; ++i){
        if (cluster_map[idx][i] < 0){continue;}
        out[idx][idy] += features[cluster_map[idx][i]][idy];   
    }
}
