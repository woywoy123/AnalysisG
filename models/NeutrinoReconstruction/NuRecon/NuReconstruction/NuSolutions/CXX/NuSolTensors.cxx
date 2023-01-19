#include "../Headers/NuSolTensors.h"
#include "../../Physics/Tensors/Headers/PhysicsTensors.h"

torch::Tensor NuSolutionTensors::x0Polar(torch::Tensor PolarL, torch::Tensor MassH, torch::Tensor MassL)
{
	return -(MassH.pow(2) 
			- MassL.pow(2) 
			- PhysicsTensors::Mass2Polar(PolarL)
		)/(2*PhysicsTensors::Slicer(PolarL, 3, 4)); 
}

torch::Tensor NuSolutionTensors::x0Cartesian(torch::Tensor CartesianL, torch::Tensor MassH, torch::Tensor MassL)
{
	return -(MassH.pow(2) 
			- MassL.pow(2) 
			- PhysicsTensors::Mass2Cartesian(CartesianL)
		)/(2*PhysicsTensors::Slicer(CartesianL, 3, 4)); 
}
