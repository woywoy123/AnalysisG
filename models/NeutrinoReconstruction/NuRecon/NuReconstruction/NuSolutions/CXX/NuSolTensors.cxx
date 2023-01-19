#include "../Headers/NuSolTensors.h"
#include "../../Physics/Tensors/Headers/PhysicsTensors.h"

torch::Tensor NuSolutionTensors::x0Polar(torch::Tensor PolarL, torch::Tensor MassH, torch::Tensor MassL)
{
	return -(MassH.pow(2) 
			- MassL.pow(2) 
			- PhysicsTensors::Mass2Polar(PolarL)
		)/(2*PhysicsTensors::Slicer(PolarL, 3, 4)); 
}

torch::Tensor NuSolutionTensors::x0Cartesian(torch::Tensor CartesianL, torch::Tensor MassH, torch::Tensor MassL)
{
	return -(MassH.pow(2) 
			- MassL.pow(2) 
			- PhysicsTensors::Mass2Cartesian(CartesianL)
		)/(2*PhysicsTensors::Slicer(CartesianL, 3, 4)); 
}

torch::Tensor NuSolutionTensors::SxCartesian(torch::Tensor _b, torch::Tensor _mu, torch::Tensor mW, torch::Tensor mNu)
{
	torch::Tensor beta_muon = PhysicsTensors::BetaCartesian(_mu);
	torch::Tensor beta_muon2 = beta_muon.pow(2); 
	torch::Tensor x0 = NuSolutionTensors::x0Cartesian(_mu, mW, mNu); 

	return (x0 * beta_muon - PhysicsTensors::PCartesian(_mu)*(1 - beta_muon2))/(beta_muon2); 

}

torch::Tensor NuSolutionTensors::SyCartesian(torch::Tensor _b, torch::Tensor _mu, 
		torch::Tensor mTop, torch::Tensor mW, torch::Tensor Sx)
{
	torch::Tensor beta_b = PhysicsTensors::BetaCartesian(_b); 
	torch::Tensor costheta = PhysicsTensors::CosThetaCartesian(_b, _mu); 
	torch::Tensor sintheta = torch::sqrt(1 - costheta.pow(2)); 
	
	torch::Tensor x0 = NuSolutionTensors::x0Cartesian(_b, mTop, mW); 

	return ((x0 / beta_b) - costheta * Sx) / sintheta; 
}

torch::Tensor NuSolutionTensors::SxSyCartesian(torch::Tensor _b, torch::Tensor _mu, 
		torch::Tensor mTop, torch::Tensor mW, torch::Tensor mNu)
{
	torch::Tensor beta_b = PhysicsTensors::BetaCartesian(_b); 
	torch::Tensor beta_muon = PhysicsTensors::BetaCartesian(_mu);
	torch::Tensor beta_muon2 = beta_muon.pow(2); 
	
	torch::Tensor costheta = PhysicsTensors::CosThetaCartesian(_b, _mu); 
	torch::Tensor sintheta = torch::sqrt(1 - costheta.pow(2)); 

	torch::Tensor x0t = NuSolutionTensors::x0Cartesian(_b, mTop, mW);
	torch::Tensor x0m = NuSolutionTensors::x0Cartesian(_mu, mW, mNu);
	       
	torch::Tensor Sx = (x0m * beta_muon - PhysicsTensors::PCartesian(_mu)*(1 - beta_muon2)) / beta_muon2; 
	torch::Tensor Sy = ((x0t / beta_b) - costheta * Sx) / sintheta;
	return torch::cat({Sx, Sy}, 1);
}

torch::Tensor NuSolutionTensors::Eps2Polar(torch::Tensor mW, torch::Tensor mNu, torch::Tensor mu)
{
	return NuSolutionTensors::Eps2Cartesian(mW, mNu, PhysicsTensors::ToPxPyPzE(mu)); 
}

torch::Tensor NuSolutionTensors::Eps2Cartesian(torch::Tensor mW, torch::Tensor mNu, torch::Tensor mu)
{
	return (mW.pow(2) - mNu.pow(2)) * (1 - PhysicsTensors::Beta2Cartesian(mu)); 
}

torch::Tensor NuSolutionTensors::wCartesian(torch::Tensor _b, torch::Tensor _mu, int factor)
{
	torch::Tensor beta_b = PhysicsTensors::BetaCartesian(_b); 
	torch::Tensor beta_mu = PhysicsTensors::BetaCartesian(_mu);

	torch::Tensor costheta = PhysicsTensors::CosThetaCartesian(_b, _mu); 
	torch::Tensor sintheta = torch::sqrt(1 - costheta.pow(2)); 
	
	return ( factor * (beta_mu / beta_b) - costheta ) / sintheta;
}

torch::Tensor NuSolutionTensors::wPolar(torch::Tensor _b, torch::Tensor _mu, int factor)
{
	return NuSolutionTensors::wCartesian(PhysicsTensors::ToPxPyPzE(_b), PhysicsTensors::ToPxPyPzE(_mu), factor); 
}

torch::Tensor NuSolutionTensors::Omega2Cartesian(torch::Tensor _b, torch::Tensor _mu)
{
	return NuSolutionTensors::wCartesian(_b, _mu, 1) + 1 - PhysicsTensors::Beta2Cartesian(_mu);  
}

torch::Tensor NuSolutionTensors::Omega2Polar(torch::Tensor _b, torch::Tensor _mu)
{
	return NuSolutionTensors::Omega2Cartesian(_b, _mu); 
}
