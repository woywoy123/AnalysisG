#include <torch/torch.h>
#include "physics.cu"

template <typename scalar_t>
__global__ void Px2Py2Pz2K(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int length)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 

    if (idy >= 3 || idx >= length){ return; }
    p2(pmc[idx][idy], pmc[idx][idy]); 
}
