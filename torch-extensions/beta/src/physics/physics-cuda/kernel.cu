#include <torch/torch.h>
#include "physics.cu"

template <typename scalar_t>
__global__ void Px2Py2Pz2K(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int length, 
        const unsigned int dims)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 

    if (idy >= dims || idx >= length){ return; }
    p2(pmc[idx][idy], pmc[idx][idy]); 
}

template <typename scalar_t> 
__global__ void SumK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        const unsigned int length, 
        const unsigned int dim_len)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= length){ return; }
    for (unsigned int x(1); x < dim_len-1; ++x)
    { 
        sum(pmc[idx][0], pmc[idx][x], pmc[idx][x+1]); 
    }
}

template <typename scalar_t>
__global__ void Div_ij_K(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> div, 
        const unsigned int length, 
        const unsigned int i, 
        const unsigned int j)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= length){ return; }
    _div_ij(pmc[idx][i], div[idx][j]);  
}

template <typename scalar_t>
__global__ void SqrtK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        const unsigned int length,
        const unsigned int dim_min,
        const unsigned int dim_max)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y + dim_min;
    if (idx >= length || dim_max < idy){ return; }   
    _sqrt(pmc[idx][idy], pmc[idx][idy]);  
}
