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
