#include "../../Tensors/Headers/PhysicsTensors.h"
#include "../Headers/PhysicsFloats.h"

torch::Tensor PhysicsFloats::ToPx(float pt, float phi, std::string device)
{
	return PhysicsFloats::ToTensor(pt*std::cos(phi), device); 
}

torch::Tensor PhysicsFloats::ToPy(float pt, float phi, std::string device)
{
	return PhysicsFloats::ToTensor(pt*std::sin(phi), device); 
}

torch::Tensor PhysicsFloats::ToPz(float pt, float eta, std::string device)
{
	return PhysicsFloats::ToTensor(pt*std::sinh(eta), device); 
}

torch::Tensor PhysicsFloats::ToPxPyPzE(float pt, float eta, float phi, float E, std::string device)
{
	return torch::cat({
				ToPx(pt, phi, device), 
				ToPy(pt, phi, device), 
				ToPz(pt, eta, device), 
				PhysicsFloats::ToTensor(E, device)
			}).view({-1, 4}); 
}

torch::Tensor PhysicsFloats::Mass2Cartesian(float px, float py, float pz, float e, std::string device)
{
	torch::Tensor Pmu = torch::tensor({px, py, pz, e}, PhysicsFloats::Options(device)).view({-1, 4}); 
	return PhysicsTensors::Mass2Cartesian(Pmu); 	
}

torch::Tensor PhysicsFloats::MassCartesian(float pt, float eta, float phi, float e, std::string device)
{
	return PhysicsFloats::Mass2Polar(pt, eta, phi, e, device).sqrt(); 	
}

torch::Tensor PhysicsFloats::P2Cartesian(float px, float py, float pz, std::string device)
{
	return torch::tensor({px, py, pz}, PhysicsFloats::Options(device)).pow(2).sum({-1}).view({-1, 1});
}

torch::Tensor PhysicsFloats::PCartesian(float px, float py, float pz, std::string device)
{
	return torch::sqrt(PhysicsFloats::P2Cartesian(px, py, pz, device));
}


torch::Tensor PhysicsFloats::BetaCartesian(float px, float py, float pz, float e, std::string device)
{
	return torch::sqrt(PhysicsFloats::P2Cartesian(px, py, pz, device))/PhysicsFloats::ToTensor(e, device);  
}

torch::Tensor PhysicsFloats::CosThetaCartesian(float px1, float px2, float py1, float py2, 
					       float pz1, float pz2, std::string device)
{
	torch::Tensor vec1 = torch::tensor({px1, py1, pz1}, PhysicsFloats::Options(device)).view({-1, 3});
	torch::Tensor vec2 = torch::tensor({px2, py2, pz2}, PhysicsFloats::Options(device)).view({-1, 3});
	return PhysicsTensors::CosThetaCartesian(vec1, vec2);
}

torch::Tensor PhysicsFloats::SinThetaCartesian(float px1, float px2, float py1, float py2, 
		                               float pz1, float pz2, std::string device)
{
	torch::Tensor vec1 = torch::tensor({px1, py1, pz1}, PhysicsFloats::Options(device)).view({-1, 3});
	torch::Tensor vec2 = torch::tensor({px2, py2, pz2}, PhysicsFloats::Options(device)).view({-1, 3});
	return PhysicsTensors::SinThetaCartesian(vec1, vec2);
}
