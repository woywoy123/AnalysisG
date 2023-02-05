#include "Operators.cu"

template <typename scalar_t> 
__global__ void _Dot2K(
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> v1, 
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> v2, 
		torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> _out, 
		const int len, const int dim)
{
	
	const int indx = blockIdx.x*blockDim.x + threadIdx.x; 
	const int indy = blockIdx.y; 
	
	if (indx >= len || indy >= dim){return;}
	_v1xv2(&_out[indx][indy], &v1[indx][indy], &v2[indx][indy]); 
}

template <typename scalar_t> 
__global__ void _CosThetaK(
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> _v12, 
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> _v22, 
		torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> _V1V2, 
		const int x)
{	
	const int indx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (indx >= x){return;}
	_costheta(&_V1V2[indx][0], &_v12[indx][0], &_v22[indx][0], &_V1V2[indx][0]); 
}

template <typename scalar_t>
__global__ void _RxK(
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> agl, 
		torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> _out)
{
	const int indx = blockIdx.x * blockDim.x + threadIdx.x; 
	const int indy = blockIdx.y; 
	const int indz = blockIdx.z; 

	if (indx >= agl.size(0) || indy >= 3 || indz >= 3){ return; }
	
	if (indy == indz && indz == 0){ _out[indx][indy][indz] = 1; return; }	
	if (indy == indz && indz > 0){ _out[indx][indy][indz] = _cos(agl[indx][0]); return; }
	if (indy == 1 && indz == 2){ _out[indx][indy][indz] = -_sin(agl[indx][0]); return; }
	if (indy == 2 && indz == 1){ _out[indx][indy][indz] = _sin(agl[indx][0]); return; }
}

template <typename scalar_t>
__global__ void _RyK(
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> agl, 
		torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> _out)
{
	const int indx = blockIdx.x * blockDim.x + threadIdx.x; 
	const int indy = blockIdx.y; 
	const int indz = blockIdx.z; 

	if (indx >= agl.size(0) || indy >= 3 || indz >= 3){ return; }

	if (indy == indz && ( indz == 0 || indz == 2) ){ _out[indx][indy][indz] = _cos(agl[indx][0]); return; }	
	if (indy == 0 && indz == 2){ _out[indx][indy][indz] = _sin(agl[indx][0]); return; }	
	if (indy == 2 && indz == 0){ _out[indx][indy][indz] = -_sin(agl[indx][0]); return; }	
	if (indy == indz && indz == 1){ _out[indx][indy][indz] = 1; return; }	
}

template <typename scalar_t>
__global__ void _RzK(
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> agl, 
		torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> _out)
{
	const int indx = blockIdx.x * blockDim.x + threadIdx.x; 
	const int indy = blockIdx.y; 
	const int indz = blockIdx.z; 

	if (indx >= agl.size(0) || indy >= 3 || indz >= 3){ return; }

	if (indy == indz && indz < 2){ _out[indx][indy][indz] = _cos(agl[indx][0]); return; }	
	if (indy == 1 && indz == 0){ _out[indx][indy][indz] = _sin(agl[indx][0]); return; }
	if (indy == 0 && indz == 1){ _out[indx][indy][indz] = -_sin(agl[indx][0]); return; }
	if (indy == indz && indz == 2){ _out[indx][indy][indz] = 1; return; }	
}
