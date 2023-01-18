#include "../Headers/NuSolTensor.h"
#include "../../BaseFunctions/Headers/PhysicsTensors.h"

torch::Tensor NuSolutionTensors::x0(torch::Tensor PolarL, torch::Tensor MassH, torch::Tensor MassL)
{
	return -(MassH.pow(2) - MassL.pow(2) - PhysicsTensors::Mass2Polar(PolarL))/(2*PhysicsTensors::Slicer(PolarL, 3, 4)); 
}
