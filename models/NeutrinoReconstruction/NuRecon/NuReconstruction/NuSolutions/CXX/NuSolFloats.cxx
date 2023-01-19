#include "../Headers/NuSolFloats.h"
#include "../../Physics/Floats/Headers/PhysicsFloats.h"

torch::Tensor NuSolutionFloats::x0Polar(float LPt, float LEta, float LPhi, float LE, 
		float MassH, float MassL, std::string device)
{
	return -(PhysicsFloats::ToTensor(MassH, device).pow(2) 
			- PhysicsFloats::ToTensor(MassL, device).pow(2) 
			- PhysicsFloats::Mass2Polar(LPt, LEta, LPhi, LE, device)
		)/(2*PhysicsFloats::ToTensor(LE, device)); 
}

torch::Tensor NuSolutionFloats::x0Cartesian(float LPx, float LPy, float LPz, float LE, 
		float MassH, float MassL, std::string device)
{
	return -(PhysicsFloats::ToTensor(MassH, device).pow(2) 
			- PhysicsFloats::ToTensor(MassL, device).pow(2) 
			- PhysicsFloats::Mass2Cartesian(LPx, LPy, LPz, LE, device)
		)/(2*PhysicsFloats::ToTensor(LE, device)); 
}


