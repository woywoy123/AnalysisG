#ifndef H_PHYSICS_CUDA
#define H_PHYSICS_CUDA

#include <torch/extension.h>
torch::Tensor _P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
torch::Tensor _P(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 

torch::Tensor _Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
torch::Tensor _Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 

torch::Tensor _M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
torch::Tensor _M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 

torch::Tensor _Mt2(torch::Tensor pz, torch::Tensor e); 
torch::Tensor _Mt(torch::Tensor pz, torch::Tensor e); 

torch::Tensor _Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz);
torch::Tensor _DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2); 


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace PhysicsCUDA
{
	const torch::Tensor P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
	{
		CHECK_INPUT(px); 
		CHECK_INPUT(py); 
		CHECK_INPUT(pz); 

		return _P2(px, py, pz);
	}

	const torch::Tensor P(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
	{
		CHECK_INPUT(px); 
		CHECK_INPUT(py); 
		CHECK_INPUT(pz); 

		return _P(px, py, pz);
	}

	const torch::Tensor Beta2(
			torch::Tensor px, torch::Tensor py, 
			torch::Tensor pz, torch::Tensor e)
	{
		CHECK_INPUT(px); 
		CHECK_INPUT(py); 
		CHECK_INPUT(pz);
		CHECK_INPUT(e); 
		return _Beta2(px, py, pz, e);
	}

	const torch::Tensor Beta(
			torch::Tensor px, torch::Tensor py, 
			torch::Tensor pz, torch::Tensor e)
	{
		CHECK_INPUT(px); 
		CHECK_INPUT(py); 
		CHECK_INPUT(pz);
		CHECK_INPUT(e); 
		return _Beta(px, py, pz, e);
	}

	const torch::Tensor M2(
			torch::Tensor px, torch::Tensor py, 
			torch::Tensor pz, torch::Tensor e)
	{
		CHECK_INPUT(px); 
		CHECK_INPUT(py); 
		CHECK_INPUT(pz);
		CHECK_INPUT(e);	
		return _M2(px, py, pz, e);
	}


	const torch::Tensor M(
			torch::Tensor px, torch::Tensor py, 
			torch::Tensor pz, torch::Tensor e)
	{
		CHECK_INPUT(px); 
		CHECK_INPUT(py); 
		CHECK_INPUT(pz);
		CHECK_INPUT(e);	
		return _M(px, py, pz, e);
	}

	const torch::Tensor Mt2(torch::Tensor pz, torch::Tensor e)
	{
		CHECK_INPUT(pz);
		CHECK_INPUT(e);	
		return _Mt2(pz, e);
	}


	const torch::Tensor Mt(torch::Tensor pz, torch::Tensor e)
	{
		CHECK_INPUT(pz);
		CHECK_INPUT(e);	
		return _Mt(pz, e);
	}

	const torch::Tensor Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
	{
		CHECK_INPUT(px); 
		CHECK_INPUT(py); 
		CHECK_INPUT(pz);
		return _Theta(px, py, pz);
	}

	const torch::Tensor DeltaR(
			torch::Tensor eta1, torch::Tensor eta2,
			torch::Tensor phi1, torch::Tensor phi2)
	{
		CHECK_INPUT(eta1); 
		CHECK_INPUT(phi1);	
		CHECK_INPUT(eta2); 
		CHECK_INPUT(phi2);	

		return _DeltaR(eta1, eta2, phi1, phi2);
	}
}
#endif
