#ifndef H_PHYSICS_TENSORS
#define H_PHYSICS_TENSORS

#include <torch/extension.h>
#include <iostream>

namespace PhysicsTensors
{
        static torch::Tensor Slicer(torch::Tensor Vector, int sdim, int edim)
        {
        	return Vector.index({torch::indexing::Slice(), torch::indexing::Slice(sdim, edim)});
        }

	torch::Tensor ToPxPyPzE(torch::Tensor Polar); 
	
	torch::Tensor Mass2Polar(torch::Tensor Polar); 
	torch::Tensor MassPolar(torch::Tensor Polar); 

	torch::Tensor Mass2Cartesian(torch::Tensor Cartesian);
	torch::Tensor MassCartesian(torch::Tensor Cartesian); 

	torch::Tensor P2Cartesian(torch::Tensor Cartesian);
	torch::Tensor PCartesian(torch::Tensor Cartesian);

	torch::Tensor P2Polar(torch::Tensor Polar); 
	
	torch::Tensor BetaPolar(torch::Tensor Polar); 
	torch::Tensor BetaCartesian(torch::Tensor Cartesian);

	torch::Tensor CosThetaCartesian(torch::Tensor CV1, torch::Tensor CV2);
	torch::Tensor SinThetaCartesian(torch::Tensor CV1, torch::Tensor CV2);
}
#endif

