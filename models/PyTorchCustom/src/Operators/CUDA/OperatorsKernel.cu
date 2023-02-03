#include "Operators.cu"

template <typename scalar_t> 
__global__ void _Dot2K(
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> v1, 
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> v2, 
		torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> _out, 
		const int len, const int dim)
{
	
	const int indx = blockIdx.x*blockDim.x + threadIdx.x; 
	const int indy = blockIdx.y; 
	
	if (indx >= len || indy >= dim){return;}
	_v1xv2(&_out[indx][indy], &v1[indx][indy], &v2[indx][indy]); 
}

template <typename scalar_t> 
__global__ void _CosThetaK(
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> _v12, 
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> _v22, 
		torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> _V1V2, 
		const int x)
{	
	const int indx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (indx >= x){return;}
	_costheta(&_V1V2[indx][0], &_v12[indx][0], &_v22[indx][0], &_V1V2[indx][0]); 
}

