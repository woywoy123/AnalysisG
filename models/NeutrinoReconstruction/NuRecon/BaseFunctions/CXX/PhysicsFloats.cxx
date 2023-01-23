#include "../Headers/PhysicsTensors.h"
#include "../Headers/PhysicsFloats.h"

torch::TensorOptions __Options( std::string device )
{
	torch::TensorOptions options = torch::TensorOptions();
	if (device == "cuda"){ options = options.device(torch::kCUDA); }
	return options;
}

torch::Tensor PhysicsFloats::ToPx(double pt, double phi, std::string device)
{
	return torch::tensor({pt*std::cos(phi)}, __Options(device)); 
}

torch::Tensor PhysicsFloats::ToPy(double pt, double phi, std::string device)
{
	return torch::tensor({pt*std::sin(phi)}, __Options(device)); 
}

torch::Tensor PhysicsFloats::ToPz(double pt, double eta, std::string device)
{
	return torch::tensor({pt*std::sinh(eta)}, __Options(device)); 
}

torch::Tensor PhysicsFloats::ToPxPyPzE(double pt, double eta, double phi, double E, std::string device)
{
	return torch::cat({ToPx(pt, phi, device), ToPy(pt, phi, device), ToPz(pt, eta, device), 
			torch::tensor({E}, __Options(device))}, 0).view({-1, 4}); 
}

torch::Tensor PhysicsFloats::Mass2Polar(double pt, double eta, double phi, double e, std::string device)
{
	torch::Tensor Pmu = PhysicsFloats::ToPxPyPzE(pt, eta, phi, e, device); 
	return PhysicsTensors::Mass2Cartesian(Pmu); 	
}

torch::Tensor PhysicsFloats::MassPolar(double pt, double eta, double phi, double e, std::string device)
{
	torch::Tensor Pmu = PhysicsFloats::ToPxPyPzE(pt, eta, phi, e, device); 
	return PhysicsTensors::MassCartesian(Pmu); 	
}


torch::Tensor PhysicsFloats::Mass2Cartesian(double px, double py, double pz, double e, std::string device)
{
	torch::Tensor Pmu = torch::tensor({px, py, pz, e}, __Options(device)).view({-1, 4}); 
	return PhysicsTensors::Mass2Cartesian(Pmu); 	
}

torch::Tensor PhysicsFloats::MassCartesian(double pt, double eta, double phi, double e, std::string device)
{
	return PhysicsFloats::Mass2Polar(pt, eta, phi, e, device).sqrt(); 	
}

torch::Tensor PhysicsFloats::P2Cartesian(double px, double py, double pz, std::string device)
{
	return torch::tensor({px, py, pz}, __Options(device)).pow(2).sum({-1}).view({-1, 1});
}

torch::Tensor PhysicsFloats::P2Polar(double pt, double eta, double phi, std::string device)
{
	return ToPx(pt, phi, device).pow(2) + ToPy(pt, phi, device).pow(2) + ToPz(pt, eta, device).pow(2);
}

torch::Tensor PhysicsFloats::BetaPolar(double pt, double eta, double phi, double e, std::string device)
{
	return torch::sqrt(PhysicsTensors::P2Cartesian(PhysicsFloats::ToPxPyPzE(pt, eta, phi, e, device)))/e; 
}

torch::Tensor PhysicsFloats::BetaCartesian(double px, double py, double pz, double e, std::string device)
{
	return torch::sqrt(PhysicsFloats::P2Cartesian(px, py, pz, device))/torch::tensor({e}, __Options(device)); 
}

torch::Tensor PhysicsFloats::CosThetaCartesian(double px1, double px2, double py1, double py2, 
		double pz1, double pz2, double e1,  double e2, std::string device)
{
	torch::Tensor vec1 = torch::tensor({px1, py1, pz1, e1}, __Options(device)).view({-1, 4});
	torch::Tensor vec2 = torch::tensor({px2, py2, pz2, e2}, __Options(device)).view({-1, 4});
	return PhysicsTensors::CosThetaCartesian(vec1, vec2);
}


