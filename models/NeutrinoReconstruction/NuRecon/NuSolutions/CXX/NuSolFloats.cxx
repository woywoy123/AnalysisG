#include "../Headers/NuSolFloat.h"
#include "../../BaseFunctions/Headers/PhysicsFloats.h"

torch::Tensor NuSolutionFloats::x0(float LPt, float LEta, float LPhi, float LE, 
		float MassH, float MassL, std::string device)
{
	return -(PhysicsFloats::ToTensor(MassH, device).pow(2) 
			- PhysicsFloats::ToTensor(MassL, device).pow(2) 
			- PhysicsFloats::Mass2Polar(LPt, LEta, LPhi, LE, device)
		)/(2*PhysicsFloats::ToTensor(LE, device)); 
}
