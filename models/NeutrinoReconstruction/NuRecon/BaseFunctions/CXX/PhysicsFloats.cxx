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

torch::Tensor PhysicsFloats::BetaPolar(double pt, double eta, double phi, double e, std::string device)
{
	return torch::sqrt(
		torch::pow(ToPx(pt, phi, device), 2) + 
		torch::pow(ToPy(pt, phi, device), 2) + 
		torch::pow(ToPz(pt, eta, device), 2))/e; 
}

torch::Tensor PhysicsFloats::BetaCartesian(double px, double py, double pz, double e, std::string device)
{
	return torch::sqrt(
		torch::pow(torch::tensor({px}, __Options(device)), 2) + 
		torch::pow(torch::tensor({py}, __Options(device)), 2) + 
		torch::pow(torch::tensor({pz}, __Options(device)), 2))/torch::tensor({e}, __Options(device)); 
}

torch::Tensor PhysicsFloats::CosThetaCartesian(double px1, double px2, double py1, double py2, 
		double pz1, double pz2, double e1,  double e2, std::string device)
{
	torch::Tensor vec1 = torch::tensor({px1, py1, pz1, e1}, __Options(device)).view({-1, 4});
	torch::Tensor vec2 = torch::tensor({px2, py2, pz2, e2}, __Options(device)).view({-1, 4});
	return PhysicsTensors::CosThetaCartesian(vec1, vec2);
}

