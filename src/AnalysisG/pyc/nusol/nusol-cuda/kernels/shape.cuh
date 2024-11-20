#include <torch/torch.h>

template <typename scalar_t>
__global__ void _shape_kernel(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> inpt,
        unsigned int dx, unsigned int dy, unsigned int dz, bool set
){

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 

    const bool skp = (idx >= dx) + (idy >= dy) + (idz >= dz); 
    if (skp){return;}
    const unsigned int bx = (set)*(idx < inpt.size(0))*idx; 
    out[idx][idy][idz] = inpt[bx][idy*set][idz]; 
}
