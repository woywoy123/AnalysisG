#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t _x0(
		const scalar_t H2, const scalar_t L2, 
		const scalar_t p2, const scalar_t e)
{
	return - (H2 - L2 - e*e + p2)/(2*e); 
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t _e2(
		const scalar_t mW2, const scalar_t mNu2, 
		const scalar_t muP2, const scalar_t mu_e)
{
	return (mW2 - mNu2) * (1 - muP2 / (mu_e*mu_e)); 
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t _w(
		const scalar_t muP2, const scalar_t bP2, const scalar_t mu_e, 
		const scalar_t b_e,  const scalar_t c_,  const scalar_t s_, 
		const int sign)
{
	return ( sign * sqrt(muP2/bP2)*(b_e/mu_e) - c_ ) / s_; 
}
