#ifndef H_PHYSICS_FLOATS
#define H_PHYSICS_FLOATS

#include <torch/extension.h>
#include <iostream>

namespace PhysicsFloats
{
	torch::Tensor ToPx(double pt, double phi, std::string device); 
	torch::Tensor ToPy(double pt, double phi, std::string device); 
	torch::Tensor ToPz(double pt, double eta, std::string device); 
	torch::Tensor ToPxPyPzE(double pt, double eta, double phi, double e, std::string device); 
	
	torch::Tensor Mass2Polar(double pt, double eta, double phi, double e, std::string device); 
	torch::Tensor MassPolar(double pt, double eta, double phi, double e, std::string device);
	torch::Tensor Mass2Cartesian(double px, double py, double pz, double e, std::string device);
	torch::Tensor MassCartesian(double px, double py, double pz, double e, std::string device);
	
	torch::Tensor BetaPolar(double pt, double eta, double phi, double e, std::string device);
	torch::Tensor BetaCartesian(double px, double py, double pz, double e, std::string device);
	
	torch::Tensor CosThetaCartesian(double px1, double px2, double py1, double py2, 
		                double pz1, double pz2, double e1,  double e2, std::string device); 
}
#endif
