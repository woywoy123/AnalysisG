#include "../Headers/NuSolTensor.h"

torch::Tensor SingleNuTensor::Sigma2(
		torch::Tensor Sxx, torch::Tensor Sxy, 
		torch::Tensor Syx, torch::Tensor Syy)
{
	torch::Tensor _S = torch::cat({Sxx, Sxy, Syx, Syy}, -1).view({-1, 2, 2});
	_S = torch::inverse(_S); 
	_S = torch::pad(_S, {0, 1, 0, 1}, "constant", 0);
	_S = torch::transpose(_S, 1, 2);
	return _S; 
}



torch::Tensor SingleNuTensor::Nu(
		torch::Tensor b, torch::Tensor mu, 
		torch::Tensor met, torch::Tensor phi, 
		torch::Tensor Sxx, torch::Tensor Sxy, torch::Tensor Syx, torch::Tensor Syy, 
		torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu)
{
	// Convert Polar to vectors 
	std::vector<torch::Tensor> b_P = NuSolTensors::_Format(b.view({-1, 4}), 4);
	std::vector<torch::Tensor> mu_P = NuSolTensors::_Format(mu.view({-1, 4}), 4); 
	
	// Convert Polar to PxPyPz
	std::vector<torch::Tensor> b_C = NuSolTensors::_Format(TransformTensors::PxPyPz(b_P[0], b_P[1], b_P[2]), 3); 
	std::vector<torch::Tensor> mu_C = NuSolTensors::_Format(TransformTensors::PxPyPz(mu_P[0], mu_P[1], mu_P[2]), 3); 
	
	// Some useful values
	torch::Tensor muP_ = PhysicsTensors::P(mu_C[0], mu_C[1], mu_C[2]); 
	torch::Tensor mu_e   = mu_P[3]; 
	torch::Tensor b_e = b_P[3]; 
	
	// Square the masses 
	torch::Tensor mT2 = mT.view({-1, 1}).pow(2); 
	torch::Tensor mW2 = mW.view({-1, 1}).pow(2); 
	torch::Tensor mNu2 = mNu.view({-1, 1}).pow(2);
	
	// Convert the Event MET and Phi to PxPy
	torch::Tensor MetX = TransformTensors::Px(met, phi); 
	torch::Tensor MetY = TransformTensors::Py(met, phi); 
	
	// Starting the algorithm 
	torch::Tensor sols_ = NuSolTensors::_Solutions(b_C, mu_C, b_e, mu_e, mT2, mW2, mNu2);
	torch::Tensor H_ = NuSolTensors::H_Matrix(sols_, b_C, mu_P[2], mu_C[2], muP_); 
	torch::Tensor S2_ = SingleNuTensor::Sigma2(Sxx, Sxy, Syx, Syy); 

	torch::Tensor delta_ = NuSolTensors::V0(MetX, MetY) - H_; 
	torch::Tensor X_ = torch::matmul(torch::transpose(delta_, 1, 2), S2_); 
	X_ = torch::matmul(X_, delta_).view({-1, 3, 3});

	torch::Tensor M_ = X_.matmul(NuSolTensors::Derivative(X_)); 
	M_ = M_ + torch::transpose(M_, 1, 2); 	

	return NuSolTensors::Intersections(M_, NuSolTensors::UnitCircle(M_)); 
}

