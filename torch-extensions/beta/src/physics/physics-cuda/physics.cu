#include <cuda.h>
#include <torch/torch.h>

template <typename scalar_t>
__device__ __forceinline__ void p2(scalar_t &p2, const scalar_t &p)
{
    p2 = (p) * (p);     
}

template <typename scalar_t>
__device__ __forceinline__ void sum(scalar_t &p, const scalar_t &p_1, const scalar_t &p_2)
{
   p += p_1 + p_2;  
}

template <typename scalar_t>
__device__ __forceinline__ void _sqrt(scalar_t &p, const scalar_t &p2)
{
	p = (p2 < 0) ? 0 : sqrt(p2);
}
