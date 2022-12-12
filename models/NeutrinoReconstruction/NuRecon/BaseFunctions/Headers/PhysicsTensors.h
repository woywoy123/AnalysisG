#ifndef H_PHYSICS_TENSORS
#define H_PHYSICS_TENSORS

#include <torch/extension.h>
#include <iostream>

namespace PhysicsTensors
{
	torch::Tensor ToPxPyPzE(torch::Tensor Polar); 
	
	torch::Tensor Mass2Polar(torch::Tensor Polar); 
	torch::Tensor MassPolar(torch::Tensor Polar); 
	torch::Tensor Mass2Cartesian(torch::Tensor Cartesian);
	torch::Tensor MassCartesian(torch::Tensor Cartesian); 
	
	torch::Tensor BetaPolar(torch::Tensor Polar); 
	torch::Tensor BetaCartesian(torch::Tensor Cartesian);

	torch::Tensor CosThetaCartesian(torch::Tensor CV1, torch::Tensor CV2);
}
#endif

