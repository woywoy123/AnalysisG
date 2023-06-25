#include <cuda.h>
#include <torch/torch.h>

template <typename scalar_t>
__device__ __forceinline__ void p2(scalar_t &p2, const scalar_t &p)
{
    p2 = (p) * (p);     
}

