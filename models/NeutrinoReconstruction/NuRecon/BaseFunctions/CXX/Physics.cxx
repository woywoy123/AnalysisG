#include "../Headers/Physics.h"

torch::TensorOptions __Options( std::string device )
{
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);
	if (device == "cuda"){ options = options.device(torch::kCUDA); }
	return options;
}

torch::Tensor ToPx(float pt, float phi, std::string device)
{
	return torch::tensor({pt*std::cos(phi)}, __Options(device)); 
}

torch::Tensor ToPy(float pt, float phi, std::string device)
{
	return torch::tensor({pt*std::sin(phi)}, __Options(device)); 
}

torch::Tensor ToPz(float pt, float eta, std::string device)
{
	return torch::tensor({pt*std::sinh(eta)}, __Options(device)); 
}
