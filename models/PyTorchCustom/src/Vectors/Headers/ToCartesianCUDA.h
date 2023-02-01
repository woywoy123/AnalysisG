#ifndef H_VECTOR_TOCARTESIAN_C
#define H_VECTOR_TOCARTESIAN_C

#include <torch/extension.h>
torch::Tensor _Px(torch::Tensor _pt, torch::Tensor _phi); 
torch::Tensor _Py(torch::Tensor _pt, torch::Tensor _phi);
torch::Tensor _Pz(torch::Tensor _pt, torch::Tensor _eta);
torch::Tensor _PxPyPz(torch::Tensor _pt, torch::Tensor _eta, torch::Tensor _phi);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace VectorCUDA
{
	const torch::Tensor Px(torch::Tensor pt, torch::Tensor phi)
	{
		CHECK_INPUT(pt); 
		CHECK_INPUT(phi); 

		return _Px(pt, phi); 
	}

	const torch::Tensor Py(torch::Tensor pt, torch::Tensor phi)
	{
		CHECK_INPUT(pt); 
		CHECK_INPUT(phi); 

		return _Py(pt, phi); 
	}

	const torch::Tensor Pz(torch::Tensor pt, torch::Tensor eta)
	{
		CHECK_INPUT(pt); 
		CHECK_INPUT(eta); 

		return _Pz(pt, eta); 
	}

	const torch::Tensor PxPyPz(
			torch::Tensor pt, 
			torch::Tensor eta, 
			torch::Tensor phi)
	{
		CHECK_INPUT(pt); 
		CHECK_INPUT(eta); 
		CHECK_INPUT(phi); 
		return _PxPyPz(pt, eta, phi); 
	}
}

#endif 
