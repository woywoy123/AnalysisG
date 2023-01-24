#include "../../Physics/Tensors/Headers/PhysicsTensors.h"
#include "../../Physics/Floats/Headers/PhysicsFloats.h"
#include "../Headers/Floats.h"

torch::Tensor SingleNu::Tensors::Sigma2(torch::Tensor sxx, torch::Tensor sxy, torch::Tensor syx, torch::Tensor syy)
{
	
	torch::Tensor matrix = torch::cat({sxx, sxy, syx, syy}, -1).view({-1, 2, 2});
	matrix = torch::inverse(matrix); 
	matrix = torch::pad(matrix, {0, 1, 0, 1}, "constant", 0); 
	matrix = torch::transpose(matrix, 1, 2);
	return matrix;
}

torch::Tensor SingleNu::Tensors::V0(torch::Tensor metx, torch::Tensor mety)
{
	torch::Tensor matrix = torch::cat({metx, mety}, -1).view({-1, 2}); 
	matrix = torch::pad(matrix, {0, 1, 0, 0}, "constant", 0);
	torch::Tensor _m = PhysicsTensors::Slicer(matrix, 2, 3); 	
	_m = torch::cat({_m, _m, _m +1}, -1).view({-1, 1, 3}); 
	matrix = torch::einsum("ij,ijk->ijk", {matrix, _m}); 
	return matrix; 
}

torch::Tensor SingleNu::Tensors::Init(
		torch::Tensor _b, torch::Tensor _mu, 
		torch::Tensor massTop, torch::Tensor massW, torch::Tensor massNu,
		torch::Tensor met, torch::Tensor phi, 
		torch::Tensor Sxx, torch::Tensor Sxy, torch::Tensor Syx, torch::Tensor Syy)
{
	torch::Tensor _bC = PhysicsTensors::ToPxPyPzE(_b); 
	torch::Tensor _muC = PhysicsTensors::ToPxPyPzE(_mu); 
	
	// ======== Algorithm ======= //
	torch::Tensor Sol_ = NuSolutionTensors::AnalyticalSolutionsCartesian(_bC, _muC, massTop, massW, massNu); 
	torch::Tensor S2_ = SingleNu::Tensors::Sigma2(Sxx, Sxy, Syx, Syy); 
	torch::Tensor V0_ = SingleNu::Tensors::V0Polar(met, phi); 
	torch::Tensor H_ = NuSolutionTensors::H_Algo(_b, _mu, Sol_);
	
	torch::Tensor dNu_ = V0_ - H_; 
	torch::Tensor X_ = torch::matmul( torch::transpose(dNu_, 1, 2), S2_ ).matmul(dNu_); 
	torch::Tensor D_ = NuSolutionTensors::Derivative(X_); 
	torch::Tensor M_ = D_ + D_.transpose(1, 2);
	torch::Tensor C_ = NuSolutionTensors::UnitCircle(_mu).repeat({_mu.sizes()[0], 1, 1}); 
	return NuSolutionTensors::Intersections(M_, C_, 0.0001);
}
