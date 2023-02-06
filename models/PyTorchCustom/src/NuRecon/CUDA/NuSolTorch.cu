#include "NuSolKernel.cu"

torch::Tensor _Solutions(
		torch::Tensor _muP2, torch::Tensor _bP2, 
		torch::Tensor _mu_e, torch::Tensor _b_e, 
		torch::Tensor _cos, torch::Tensor _sin,
		torch::Tensor mT2, torch::Tensor mW2, torch::Tensor mNu2)
{

	const int x = _mu_e.size(0);
	const int threads = 1024;
	
	const int y = 5; 
	const dim3 blocks((x + threads -1)/threads, y); 
	torch::Tensor out_ = torch::cat({_cos, _sin}, -1); 
	torch::Tensor x0_ = torch::zeros_like(out_); 
	torch::Tensor w_ = torch::zeros_like(out_); 
	torch::Tensor Om2_ = torch::zeros_like(out_); 

	AT_DISPATCH_FLOATING_TYPES(out_.scalar_type(), "_SolsK", ([&]
	{
		_baseValsK<scalar_t><<<blocks, threads>>>(
				_muP2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_bP2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_mu_e.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_b_e.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_cos.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
				_sin.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				mT2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				mW2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				mNu2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				x0_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				w_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				Om2_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				x, y
		); 
	})); 

	return out_;  
}
