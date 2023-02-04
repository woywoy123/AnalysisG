#ifndef H_TRANSFORM_TOPOLAR_C
#define H_TRANSFORM_TOPOLAR_C

#include <torch/extension.h>
torch::Tensor _Pt(torch::Tensor _px, torch::Tensor _py);  
torch::Tensor _Phi(torch::Tensor _px, torch::Tensor _py); 
torch::Tensor _Eta(torch::Tensor _px, torch::Tensor _py, torch::Tensor _pz);
torch::Tensor _PtEtaPhi(torch::Tensor _px, torch::Tensor _py, torch::Tensor _pz);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace TransformCUDA
{
	const torch::Tensor PT(torch::Tensor px, torch::Tensor py)
	{
		CHECK_INPUT(px); 
		CHECK_INPUT(py); 

		return _Pt(px, py); 
	}

	const torch::Tensor Phi(torch::Tensor px, torch::Tensor py)
	{
		CHECK_INPUT(px); 
		CHECK_INPUT(py); 

		return _Phi(px, py); 
	}

	const torch::Tensor Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
	{
		CHECK_INPUT(px); 
		CHECK_INPUT(py); 
		CHECK_INPUT(pz); 

		return _Eta(px, py, pz);  
	}

	const torch::Tensor PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
	{
		CHECK_INPUT(px); 
		CHECK_INPUT(py); 
		CHECK_INPUT(pz); 
		return _PtEtaPhi(px, py, pz); 
	}
}

#endif 
