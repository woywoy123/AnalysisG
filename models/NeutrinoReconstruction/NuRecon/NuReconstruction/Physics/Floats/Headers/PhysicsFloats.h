#ifndef H_PHYSICS_FLOATS
#define H_PHYSICS_FLOATS

#include <torch/extension.h>
#include <iostream>

namespace PhysicsFloats
{
	static torch::TensorOptions Options( std::string device )
	{
		torch::TensorOptions options = torch::TensorOptions();
		if (device == "cuda"){ options = options.device(torch::kCUDA); }
		return options;
	}

	static torch::Tensor ToTensor(float var, std::string device)
	{
		return torch::tensor({var}, PhysicsFloats::Options(device)).view({-1, 1}); 	
	}

	static torch::Tensor ToTensor(float px, float py, float pz, float e, std::string device)
	{
	 	return torch::tensor({px, py, pz, e}, PhysicsFloats::Options(device)).view({-1, 4});
	}

	torch::Tensor ToPx(float pt, float phi, std::string device); 
	torch::Tensor ToPy(float pt, float phi, std::string device); 
	torch::Tensor ToPz(float pt, float eta, std::string device); 
	torch::Tensor ToPxPyPzE(float pt, float eta, float phi, float e, std::string device); 
	
	torch::Tensor Mass2Polar(float pt, float eta, float phi, float e, std::string device); 
	torch::Tensor MassPolar(float pt, float eta, float phi, float e, std::string device);

	torch::Tensor Mass2Cartesian(float px, float py, float pz, float e, std::string device);
	torch::Tensor MassCartesian(float px, float py, float pz, float e, std::string device);
	
	torch::Tensor P2Cartesian(float px, float py, float pz, std::string device);
	torch::Tensor P2Polar(float pt, float eta, float phi, std::string device); 

	torch::Tensor PCartesian(float px, float py, float pz, std::string device);
	torch::Tensor PPolar(float pt, float eta, float phi, std::string device); 

	torch::Tensor BetaPolar(float pt, float eta, float phi, float e, std::string device);
	torch::Tensor BetaCartesian(float px, float py, float pz, float e, std::string device);

	torch::Tensor Beta2Polar(float pt, float eta, float phi, float e, std::string device);
	torch::Tensor Beta2Cartesian(float px, float py, float pz, float e, std::string device);
	

	torch::Tensor CosThetaCartesian(float px1, float px2, 
			float py1, float py2, float pz1, float pz2, std::string device); 
	torch::Tensor SinThetaCartesian(float px1, float px2, 
			float py1, float py2, float pz1, float pz2, std::string device);
}
#endif
