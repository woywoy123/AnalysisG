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
	return (mW - mNu) * (1 - PhysicsTensors::Beta2Cartesian(mu)); 
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
	return NuSolutionTensors::wCartesian(_b, _mu, 1).pow(2) + 1 - PhysicsTensors::Beta2Cartesian(_mu);  
}

torch::Tensor NuSolutionTensors::Omega2Polar(torch::Tensor _b, torch::Tensor _mu)
{
	return NuSolutionTensors::Omega2Cartesian(PhysicsTensors::ToPxPyPzE(_b), PhysicsTensors::ToPxPyPzE(_mu)); 
}

torch::Tensor NuSolutionTensors::AnalyticalSolutionsCartesian(torch::Tensor _b, torch::Tensor _mu,
		torch::Tensor massTop, torch::Tensor massW, torch::Tensor massNu)
{
	// ====================== Muon =================================== //
	// Slice the given vector into rows 
	torch::Tensor mu_px = PhysicsTensors::Slicer(_mu, 0, 1); 
	torch::Tensor mu_py = PhysicsTensors::Slicer(_mu, 1, 2); 
	torch::Tensor mu_pz = PhysicsTensors::Slicer(_mu, 2, 3); 
	torch::Tensor mu_e  = PhysicsTensors::Slicer(_mu, 3, 4); 
	
	// Square the momentum components
	_mu = _mu.pow(2); 
	torch::Tensor mu_p2x = PhysicsTensors::Slicer(_mu, 0, 1); 
	torch::Tensor mu_p2y = PhysicsTensors::Slicer(_mu, 1, 2); 
	torch::Tensor mu_p2z = PhysicsTensors::Slicer(_mu, 2, 3); 
	torch::Tensor mu_e2  = PhysicsTensors::Slicer(_mu, 3, 4);
	
	// Get additional kinematic variables P2, mass2, beta 
	torch::Tensor mu_P2    = mu_p2x + mu_p2y + mu_p2z;
	torch::Tensor mu_mass2 = mu_e2 - mu_P2; 
	torch::Tensor mu_beta2 = mu_P2 / mu_e2;

	// ====================== b-quark =================================== //
	// Slice the given vector into rows 
	torch::Tensor b_px = PhysicsTensors::Slicer(_b, 0, 1); 
	torch::Tensor b_py = PhysicsTensors::Slicer(_b, 1, 2); 
	torch::Tensor b_pz = PhysicsTensors::Slicer(_b, 2, 3); 
	torch::Tensor b_e  = PhysicsTensors::Slicer(_b, 3, 4); 
	
	// Square the momentum components
	_b = _b.pow(2); 
	torch::Tensor b_p2x = PhysicsTensors::Slicer(_b, 0, 1); 
	torch::Tensor b_p2y = PhysicsTensors::Slicer(_b, 1, 2); 
	torch::Tensor b_p2z = PhysicsTensors::Slicer(_b, 2, 3); 
	torch::Tensor b_e2  = PhysicsTensors::Slicer(_b, 3, 4);
	
	// Get additional kinematic variables P2, mass2, beta 
	torch::Tensor b_P2    = b_p2x + b_p2y + b_p2z;
	torch::Tensor b_mass2 = b_e2 - b_P2; 
	torch::Tensor b_beta = torch::sqrt(b_P2 / b_e2);	
	
	// ===================== costheta ==================== //
	torch::Tensor costheta = (b_px * mu_px + b_py * mu_py + b_pz * mu_pz) / torch::sqrt(mu_P2 * b_P2); 
	torch::Tensor sintheta = torch::sqrt(1 - costheta.pow(2)); 

	// ===================== masses ===================== //
	massTop = massTop.pow(2); 
	massW = massW.pow(2); 
	massNu = massNu.pow(2);
	torch::Tensor _r = torch::sqrt(mu_beta2) / b_beta; 

	// ===================== Algo Variables ============= //
	torch::Tensor x0p = - ( massTop - massW - b_mass2) / (2 * b_e); 
	torch::Tensor x0  = - ( massW - mu_mass2 - massNu) / (2 * mu_e); 
	torch::Tensor Sx = (x0 * torch::sqrt(mu_beta2) - torch::sqrt(mu_P2) * (1 - mu_beta2)) / mu_beta2; 
	torch::Tensor Sy = ((x0p / b_beta) - costheta * Sx) / sintheta; 
	torch::Tensor eps2 = (massW - massNu) * (1 - mu_beta2); 
	torch::Tensor w = ( _r - costheta ) / sintheta;
	torch::Tensor w_ = (-_r - costheta) / sintheta;
	torch::Tensor Omega2 = w.pow(2) + 1 - mu_beta2; 
	_r = Sx + w * Sy; 
	torch::Tensor x = Sx - (_r) / Omega2;
	torch::Tensor y = Sy - (_r) * w / Omega2; 
	torch::Tensor z2 = x.pow(2) * Omega2 - (Sy - w * Sx).pow(2) - (massW - x0.pow(2) - eps2); 
	
	return torch::cat({costheta, sintheta, x0, x0p, Sx, Sy, w, w_, x, y, z2, Omega2, eps2}, 1); 
}

torch::Tensor NuSolutionTensors::AnalyticalSolutionsPolar(torch::Tensor _b, torch::Tensor _mu,
		torch::Tensor massTop, torch::Tensor massW, torch::Tensor massNu)
{
	return NuSolutionTensors::AnalyticalSolutionsCartesian(
			PhysicsTensors::ToPxPyPzE(_b), 
			PhysicsTensors::ToPxPyPzE(_mu), 
			massTop, massW, massNu); 
}
