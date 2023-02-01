#include "../Headers/ToCartesianTensors.h"

torch::Tensor VectorTensors::Px(torch::Tensor pt, torch::Tensor phi)
{
	return pt * torch::cos(phi);
}

torch::Tensor VectorTensors::Py(torch::Tensor pt, torch::Tensor phi)
{
	return pt * torch::sin(phi);
}

torch::Tensor VectorTensors::Pz(torch::Tensor pt, torch::Tensor eta)
{
	return pt * torch::sinh(eta);
}

torch::Tensor VectorTensors::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi)
{
	torch::Tensor _px = Px(pt, phi).view({-1, 1});
	torch::Tensor _py = Py(pt, phi).view({-1, 1}); 
	torch::Tensor _pz = Pz(pt, eta).view({-1, 1}); 
	return torch::cat({_px, _py, _pz}, -1);
}
