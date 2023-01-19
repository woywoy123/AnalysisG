#include "../../Tensors/Headers/PhysicsTensors.h"
#include "../Headers/PhysicsFloats.h"

torch::Tensor PhysicsFloats::Mass2Polar(float pt, float eta, float phi, float e, std::string device)
{
	torch::Tensor Pmu = PhysicsFloats::ToPxPyPzE(pt, eta, phi, e, device); 
	return PhysicsTensors::Mass2Cartesian(Pmu); 	
}

torch::Tensor PhysicsFloats::MassPolar(float pt, float eta, float phi, float e, std::string device)
{
	torch::Tensor Pmu = PhysicsFloats::ToPxPyPzE(pt, eta, phi, e, device); 
	return PhysicsTensors::MassCartesian(Pmu); 	
}

torch::Tensor PhysicsFloats::P2Polar(float pt, float eta, float phi, std::string device)
{
	return PhysicsFloats::ToPx(pt, phi, device).pow(2) + 
		PhysicsFloats::ToPy(pt, phi, device).pow(2) + 
		PhysicsFloats::ToPz(pt, eta, device).pow(2);
}

torch::Tensor PhysicsFloats::PPolar(float pt, float eta, float phi, std::string device)
{
	return torch::sqrt(PhysicsFloats::P2Polar(pt, eta, phi, device)); 
}

torch::Tensor PhysicsFloats::BetaPolar(float pt, float eta, float phi, float e, std::string device)
{
	return torch::sqrt(PhysicsTensors::P2Cartesian(PhysicsFloats::ToPxPyPzE(pt, eta, phi, e, device)))/e; 
}


