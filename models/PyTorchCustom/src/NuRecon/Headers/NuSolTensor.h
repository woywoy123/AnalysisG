#ifndef H_NUSOL_TENSOR
#define H_NUSOL_TENSOR

#include <torch/extension.h>
#include "../../Physics/Headers/Tensors.h"
#include "../../Transform/Headers/ToCartesianTensors.h"
#include "../../Operators/Headers/Tensors.h"

namespace NuSolTensors
{
	torch::Tensor _Solutions(
		torch::Tensor b_pt, torch::Tensor b_eta, torch::Tensor b_phi, torch::Tensor b_e,
		torch::Tensor b_px, torch::Tensor b_py, torch::Tensor b_pz,
		torch::Tensor mu_pt, torch::Tensor mu_eta, torch::Tensor mu_phi, torch::Tensor mu_e,
		torch::Tensor mu_px, torch::Tensor mu_py, torch::Tensor mu_pz,
		torch::Tensor massT, torch::Tensor massW, torch::Tensor massNu); 
	
	std::vector<torch::Tensor> _Format(torch::Tensor t, int dim);
	torch::Tensor x0(torch::Tensor _pe, torch::Tensor _pm2, torch::Tensor mH2, torch::Tensor mL2); 
	
	torch::Tensor H_Matrix(
			torch::Tensor Sols_, 
			torch::Tensor b_px, torch::Tensor b_py, torch::Tensor b_pz, 
			torch::Tensor mu_phi, torch::Tensor mu_pz, torch::Tensor mu_P); 

	torch::Tensor Solutions(torch::Tensor b, torch::Tensor mu, torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu);
}

namespace SingleNuTensor
{
	torch::Tensor Nu(torch::Tensor b, torch::Tensor mu, torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu);
}

#endif 
