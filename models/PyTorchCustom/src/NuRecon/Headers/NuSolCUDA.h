#ifndef H_NUSOL_CUDA
#define H_NUSOL_CUDA 

#include <torch/extension.h>
#include <iostream>
#include "../../Physics/Headers/CUDA.h"
#include "../../Transform/Headers/ToCartesianCUDA.h"
#include "../../Operators/Headers/CUDA.h"

torch::Tensor _Solutions(
		torch::Tensor _muP2, torch::Tensor _bP2, 
		torch::Tensor _mu_e, torch::Tensor _b_e, 
		torch::Tensor _cos, torch::Tensor _sin, 
		torch::Tensor mT2, torch::Tensor mW2, torch::Tensor mNu2); 

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


namespace NuSolCUDA
{
	const void _CheckTensors(std::vector<torch::Tensor> T){for (torch::Tensor x : T){CHECK_INPUT(x);}}

	const torch::Tensor Solutions(
		std::vector<torch::Tensor> b_P, std::vector<torch::Tensor> b_C, 
		std::vector<torch::Tensor> mu_P, std::vector<torch::Tensor> mu_C, 
		torch::Tensor massT2, torch::Tensor massW2, torch::Tensor massNu2)
	{
		_CheckTensors({massT2, massW2, massNu2});	
		torch::Tensor _muP2 = PhysicsCUDA::P2(mu_C[0], mu_C[1], mu_C[2]);	
		torch::Tensor _bP2 = PhysicsCUDA::P2(b_C[0], b_C[1], b_C[2]); 

		torch::Tensor _cos = OperatorsCUDA::CosTheta(
				torch::cat({b_C[0], b_C[1], b_C[2]}, -1), 
				torch::cat({mu_C[0], mu_C[1], mu_C[2]}, -1)
		);

		torch::Tensor _sin = OperatorsCUDA::_SinTheta(_cos); 

		return _Solutions(_muP2, _bP2, mu_C[3], b_C[3], _cos, _sin, massT2, massW2, massNu2);
	}
	
	const torch::Tensor H_Matrix(
			torch::Tensor Sols_, 
			std::vector<torch::Tensor> b_C, 
			torch::Tensor mu_phi, torch::Tensor mu_pz, torch::Tensor mu_P)
	{
		return mu_P; 

	}

	const std::vector<torch::Tensor> _Format(torch::Tensor t, int dim)
	{
		std::vector<torch::Tensor> _out; 
		for (int i = 0; i < dim; ++i)
		{
			_out.push_back((t.index({torch::indexing::Slice(), i}).view({-1, 1}).contiguous())); 
		}
		return _out; 
	}
}

namespace SingleNuCUDA
{
	const torch::Tensor Nu(torch::Tensor b, torch::Tensor mu, torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu)
	{
		// ---- Polar Version of Particles ---- //
		std::vector<torch::Tensor> b_P = NuSolCUDA::_Format(b.view({-1, 4}), 4);
		std::vector<torch::Tensor> mu_P = NuSolCUDA::_Format(mu.view({-1, 4}), 4); 
		
		// ---- Cartesian Version of Particles ---- //
		std::vector<torch::Tensor> b_C = NuSolCUDA::_Format(TransformCUDA::PxPyPz(b_P[0], b_P[1], b_P[2]), 3); 
		std::vector<torch::Tensor> mu_C = NuSolCUDA::_Format(TransformCUDA::PxPyPz(mu_P[0], mu_P[1], mu_P[2]), 3); 
		torch::Tensor muP_ = PhysicsCUDA::P(mu_C[0], mu_C[1], mu_C[2]); 
		
		// ---- Precalculate the Mass Squared ---- //
		torch::Tensor mT2 = OperatorsCUDA::Dot(mT, mT); 
		torch::Tensor mW2 = OperatorsCUDA::Dot(mW, mW); 
		torch::Tensor mNu2 = OperatorsCUDA::Dot(mNu, mNu); 
		
		torch::Tensor sols_ = NuSolCUDA::Solutions(b_P, b_C, mu_P, mu_C, mT2, mW2, mNu2);
		torch::Tensor H_ = NuSolCUDA::H_Matrix(sols_, b_C, mu_P[2], mu_C[2], muP_); 	

		return mNu;
	}

}

#endif 
