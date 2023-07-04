#include <cuda.h>

template <typename scalar_t>
__device__ __forceinline__ void dot_ij(scalar_t &p2, const scalar_t &p)
{
    p2 *= (p);     
}

template <typename scalar_t>
__device__ __forceinline__ void dot_ij(scalar_t &o, const scalar_t &v1, const scalar_t &v2)
{
    o = v1*v2;     
}

template <typename scalar_t>
__device__ __forceinline__ void sum(scalar_t &p, const scalar_t &p_1)
{
    p += p_1; 
}
