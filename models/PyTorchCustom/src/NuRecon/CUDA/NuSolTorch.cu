#include "NuSolKernel.cu"

torch::Tensor _Solutions(
		torch::Tensor _muP2, torch::Tensor _bP2, 
		torch::Tensor _mu_e, torch::Tensor _b_e, 
		torch::Tensor _cos, torch::Tensor _sin,
		torch::Tensor mT2, torch::Tensor mW2, torch::Tensor mNu2)
{

	const int x = _mu_e.size(0);
	const int threads = 1024;
	
	const dim3 blocks1((x + threads -1)/threads, 10); 
	const dim3 blocks2((x + threads -1)/threads, 3);
	const dim3 blocks3((x + threads -1)/threads);

	torch::Tensor out_ = torch::zeros_like(_cos); 
	out_ = torch::cat({out_, out_, out_, 
			   out_, out_, out_, 
			   out_, out_, out_, 
			   out_, out_, out_, out_}, -1); 
	torch::Tensor bB_ = torch::zeros_like(_cos); 
	torch::Tensor muB_ = torch::zeros_like(_cos); 
	torch::Tensor muB2_ = torch::zeros_like(_cos); 

	//{ c_, s_, x0, x0p, Sx, Sy, w, w_, x1, y1, Z, O2, e2}
	//   0,  1,  2,   3,  4,  5, 6,  7,  8,  9,10, 11, 12

	// Calculates: bB, muB, muB2, x0p, x0, w_, w, 
	AT_DISPATCH_FLOATING_TYPES(out_.scalar_type(), "_SolsK", ([&]
	{
		_baseValsK<scalar_t><<<blocks1, threads>>>(
				_muP2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_bP2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_mu_e.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_b_e.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_cos.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
				_sin.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				mT2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				mW2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				mNu2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				out_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				bB_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				muB_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				muB2_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				x, 10
		); 
		_baseValsK<scalar_t><<<blocks2, threads>>>(
				muB2_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				muB_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_muP2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				out_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				x, 3
		); 

		_baseVals_K<scalar_t><<<blocks3, threads>>>(
				bB_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_cos.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_sin.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				out_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				x
		); 

		_baseValsK<scalar_t><<<blocks2, threads>>>(
				mW2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				out_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				x, 3
		); 

		_baseValsK<scalar_t><<<blocks3, threads>>>(
				out_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				x
		); 
	})); 
	return out_;  
}

torch::Tensor _H_Matrix(torch::Tensor sols, torch::Tensor mu_P)
{
	const int x = sols.size(0); 
	const int threads = 1024; 
	const dim3 blocks( (x + threads -1)/threads, 3, 3); 
	torch::Tensor out_ = torch::zeros(
			{x, 3, 3}, 
			torch::TensorOptions().dtype(sols.dtype()).device(sols.device())); 
	
	AT_DISPATCH_FLOATING_TYPES(out_.scalar_type(), "_H", ([&]
	{
		_HMatrix<scalar_t><<<blocks, threads>>>(
				sols.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
				mu_P.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
				out_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
				x, 3, 3
		); 
	})); 
	return out_; 
}
