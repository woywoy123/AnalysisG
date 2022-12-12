#include "../Headers/NuSolFloat.h"
#include "../Header/PhysicsFloats.h"

torch::TensorOptions __Options( std::string device )
{
	torch::TensorOptions options = torch::TensorOptions();
	if (device == "cuda"){ options = options.device(torch::kCUDA); }
	return options;
}

torch::Tensor ToTensor(double var)
{
	return torch::tensor({var}, __Options(device)); 	
}

torch::Tensor x0p(double bPt, double bEta, double bPhi, double bE, double MassTop, double MassW, std::string device)
{
	return -(ToTensor(MassTop).pow(2) - ToTensor(MassW).pow(2) - Mass2Polar(bPt, bEta, bPhi, bE))/(2*bE); 
}

torch::Tensor x0(double MuPt, double MuEta, double MuPhi, double MuE, double MassW, std::string device)
{
	return -(ToTensor(MassW).pow(2) - Mass2Polar(MuPt, MuEta, MuPhi, MuE))/(2*MuE); 
}

torch::Tensor SX(double MuPt, double MuEta, double MuPhi, double MuE, double MassW, std::string device)
{
	return (x0(MuPt, MuEta, MuPhi, MuE, device)*BetaPolar(MuPt, MuEta, MuPhi, MuE));


}
