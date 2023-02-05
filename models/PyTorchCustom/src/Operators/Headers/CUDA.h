#ifndef H_OPERATORS_CUDA
#define H_OPERATORS_CUDA

#include <torch/extension.h>

torch::Tensor _Dot(torch::Tensor v1, torch::Tensor v2);
torch::Tensor _CosTheta(torch::Tensor v1, torch::Tensor v2);
torch::Tensor _Rx(torch::Tensor angle); 
torch::Tensor _Ry(torch::Tensor angle); 
torch::Tensor _Rz(torch::Tensor angle); 

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace OperatorsCUDA
{
	const torch::Tensor Dot(torch::Tensor v1, torch::Tensor v2)
	{
		CHECK_INPUT(v1); 
		CHECK_INPUT(v2); 
		return _Dot(v1, v2).sum({-1}, true); 
	}

	const torch::Tensor CosTheta(torch::Tensor v1, torch::Tensor v2)
	{
		CHECK_INPUT(v1);
		CHECK_INPUT(v2);

		return _CosTheta(v1, v2); 
	}

	const torch::Tensor _SinTheta(torch::Tensor cos)
	{
		return torch::sqrt(1 - _Dot(cos, cos)); 
	}

	const torch::Tensor SinTheta(torch::Tensor v1, torch::Tensor v2)
	{
		CHECK_INPUT(v1);
		CHECK_INPUT(v2);
		
		return _SinTheta(_CosTheta(v1, v2)); 
	}

	const torch::Tensor Rx(torch::Tensor angle)
	{
		CHECK_INPUT(angle);
		return _Rx(angle); 
	}

	const torch::Tensor Ry(torch::Tensor angle)
	{
		CHECK_INPUT(angle);
		return _Ry(angle); 
	}

	const torch::Tensor Rz(torch::Tensor angle)
	{
		CHECK_INPUT(angle);
		return _Rz(angle); 
	}
}

#endif 
