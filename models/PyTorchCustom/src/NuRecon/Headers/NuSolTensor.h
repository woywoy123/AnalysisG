#ifndef H_NUSOL_TENSOR
#define H_NUSOL_TENSOR

#include <torch/extension.h>
#include "../../Physics/Headers/Tensors.h"
#include "../../Transform/Headers/ToCartesianTensors.h"
#include "../../Operators/Headers/Tensors.h"

namespace NuSolTensors
{
	std::vector<torch::Tensor> _Format(torch::Tensor t, int dim);
	torch::Tensor x0(torch::Tensor _pe, torch::Tensor _pm2, torch::Tensor mH2, torch::Tensor mL2); 
	torch::Tensor _UnitCircle(torch::Tensor x, std::vector<int> diag); 
	torch::Tensor UnitCircle(torch::Tensor x);
	torch::Tensor Derivative(torch::Tensor x); 
	torch::Tensor V0(torch::Tensor metx, torch::Tensor mety); 
	torch::Tensor Cofactors(torch::Tensor Matrix, int i, int j); 

	torch::Tensor H_Matrix(
			torch::Tensor Sols_, std::vector<torch::Tensor> b_C, 
			torch::Tensor mu_phi, torch::Tensor mu_pz, torch::Tensor mu_P); 

	torch::Tensor _Solutions(
		std::vector<torch::Tensor> b_C, std::vector<torch::Tensor> mu_C, 
		torch::Tensor b_e, torch::Tensor mu_e,
		torch::Tensor massT2, torch::Tensor massW2, torch::Tensor massNu2); 
	
	torch::Tensor Solutions(
			torch::Tensor b, torch::Tensor mu, 
			torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu);

	torch::Tensor HorizontalVertical(torch::Tensor G);
	torch::Tensor Parallel(torch::Tensor G); 
	torch::Tensor Intersections(torch::Tensor G, torch::Tensor g22); 
	torch::Tensor Intersecting(torch::Tensor A, torch::Tensor B); 
	
}

namespace SingleNuTensor
{
	torch::Tensor Sigma2(torch::Tensor Sxx, torch::Tensor Sxy, torch::Tensor Syx, torch::Tensor Syy); 
	torch::Tensor Nu(torch::Tensor b, torch::Tensor mu, 
			torch::Tensor met, torch::Tensor phi, 
			torch::Tensor Sxx, torch::Tensor Sxy, torch::Tensor Syx, torch::Tensor Syy, 
			torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu);
}

namespace DoubleNuTensor 
{
	torch::Tensor N(torch::Tensor H); 

	torch::Tensor NuNu(
			torch::Tensor b, torch::Tensor b_, 
			torch::Tensor mu, torch::Tensor mu_, 
			torch::Tensor met, torch::Tensor phi, 
			torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu
	); 
}





#endif 
