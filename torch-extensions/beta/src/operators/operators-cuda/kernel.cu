#include <torch/torch.h>
#include "operators.cu"

template <typename scalar_t>
__global__ void _DotK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> i, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> j, 
        const unsigned int dim_i, 
        const unsigned int dim_j)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    if (idx >= dim_i || idy >= dim_j){return;}
    dot_ij(i[idx][idy], j[idx][idy]); 
}

template <typename scalar_t>
__global__ void _DotK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v1,
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v2, 
        const unsigned int dim_z, 
        const unsigned int dim_i1,
        const unsigned int dim_co, 
        const unsigned int dim_j2)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idz >= dim_z || idy >= dim_co || idx >= dim_i1*dim_j2){return;}
    dot_ij(v1[idz][idy*dim_co+idx][idx%dim_j2], v2[idz][idy][idx%dim_j2]); 
}

template <typename scalar_t>
__global__ void _SumK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int length, 
        const unsigned int len_j)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= length){ return; }
    for (unsigned int i(1); i < len_j; ++i)
    {
        sum(pmc[idx][0], pmc[idx][i]);  
    }
}

template <typename scalar_t>
__global__ void _SumK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> mul, 
        const unsigned int dim_z,
        const unsigned int dim_x,  
        const unsigned int dim_y, 
        const unsigned int range)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idz >= dim_z || idy >= dim_y || idx >= dim_x){return;}
    
    for (unsigned int i(0); i < range; ++i)
    {
        sum(out[idz][idx][idy], mul[idz][idx][range*idy+i]);  
    }
}



