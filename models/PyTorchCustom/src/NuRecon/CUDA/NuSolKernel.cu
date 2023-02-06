#include "NuSol.cu"

template <typename scalar_t>
__global__ void _baseValsK(
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> muP2, 
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> bP2,
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mue, 
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> be,
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> cos,
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> sin, 
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mT2, 
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mW2, 
		const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mNu2,
		torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> _x0_out, 
		torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> _w_out, 
		torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> _Om2, 
		const int x, const int y)
{
	const int indx = blockIdx.x*blockDim.x + threadIdx.x; 
	const int indy = blockIdx.y;

	if (indx >= x || indy >= y){return;}
	if (indy == 0){	_x0_out[indx][0] = _x0(mT2[indx][0], mW2[indx][0], bP2[indx][0], be[indx][0]); return; }
	if (indy == 1){	_x0_out[indx][1] = _x0(mW2[indx][0], mNu2[indx][0], muP2[indx][0], mue[indx][0]); return; }
	if (indy == 2)
	{ 
		_w_out[indx][0] = _w(muP2[indx][0], bP2[indx][0], mue[indx][0], be[indx][0], cos[indx][0], sin[indx][0], 1); 
		_Om2[indx][0] = _w_out[indx][0].pow(2) + 1 - (muP2[indx][0]/mue[indx][0].pow(2)); 
		return; 
	}
	if (indy == 3)
	{ 
		_w_out[indx][1] = _w(muP2[indx][0], bP2[indx][0], mue[indx][0], be[indx][0], cos[indx][0], sin[indx][0], -1); 
		return; 
	}	

	if (indy == 4)
	{ 
		_Om2[indx][1] = _e2(mW2[indx][0], mNu2[indx][0], muP2[indx][0], mue[indx][0]); 
		return; 
	}	

}

		
