#include "../Headers/NuSolFloats.h"
#include "../Headers/NuSolTensors.h"
#include "../../Physics/Floats/Headers/PhysicsFloats.h"

torch::Tensor NuSolutionFloats::x0Polar(float LPt, float LEta, float LPhi, float LE, 
		float MassH, float MassL, std::string device)
{
	return -(PhysicsFloats::ToTensor(MassH, device).pow(2) 
			- PhysicsFloats::ToTensor(MassL, device).pow(2) 
			- PhysicsFloats::Mass2Polar(LPt, LEta, LPhi, LE, device)
		)/(2*PhysicsFloats::ToTensor(LE, device)); 
}

torch::Tensor NuSolutionFloats::x0Cartesian(float LPx, float LPy, float LPz, float LE, 
		float MassH, float MassL, std::string device)
{
	return -(PhysicsFloats::ToTensor(MassH, device).pow(2) 
			- PhysicsFloats::ToTensor(MassL, device).pow(2) 
			- PhysicsFloats::Mass2Cartesian(LPx, LPy, LPz, LE, device)
		)/(2*PhysicsFloats::ToTensor(LE, device)); 
}

torch::Tensor NuSolutionFloats::SxPolar(
		float bPt, float bEta, float bPhi, float bE, 
		float muPt, float muEta, float muPhi, float muE, 
		float mW, float mNu, std::string device)
{
	torch::Tensor _b = PhysicsFloats::ToPxPyPzE(bPt, bEta, bPhi, bE, device); 
	torch::Tensor _mu = PhysicsFloats::ToPxPyPzE(muPt, muEta, muPhi, muE, device); 

	return NuSolutionTensors::SxCartesian(_b, _mu, 
			PhysicsFloats::ToTensor(mW, device), 
			PhysicsFloats::ToTensor(mNu, device)); 
}

torch::Tensor NuSolutionFloats::SyPolar(
		float bPt, float bEta, float bPhi, float bE, 
		float muPt, float muEta, float muPhi, float muE, 
		float mTop, float mW, float mNu, std::string device)
{
	torch::Tensor _b = PhysicsFloats::ToPxPyPzE(bPt, bEta, bPhi, bE, device); 
	torch::Tensor _mu = PhysicsFloats::ToPxPyPzE(muPt, muEta, muPhi, muE, device); 

	torch::Tensor _mTop = PhysicsFloats::ToTensor(mTop, device); 
	torch::Tensor _mW = PhysicsFloats::ToTensor(mW, device); 
	torch::Tensor _mNu = PhysicsFloats::ToTensor(mNu, device); 

	torch::Tensor Sx = NuSolutionTensors::SxCartesian(_b, _mu, _mW, _mNu); 
	return NuSolutionTensors::SyCartesian(_b, _mu, _mTop, _mW, Sx); 
}

torch::Tensor NuSolutionFloats::Eps2Cartesian(float mu_px, float mu_py, float mu_pz, float mu_e, float mW, float mNu, std::string device)
{
	torch::Tensor _mu = PhysicsFloats::ToTensor(mu_px, mu_py, mu_pz, mu_e, device);
	return NuSolutionTensors::Eps2Cartesian(PhysicsFloats::ToTensor(mW, device), PhysicsFloats::ToTensor(mNu, device), _mu); 
}

torch::Tensor NuSolutionFloats::Eps2Polar(float mu_pt, float mu_eta, float mu_phi, float mu_e, float mW, float mNu, std::string device)
{
	torch::Tensor _mu = PhysicsFloats::ToPxPyPzE(mu_pt, mu_eta, mu_phi, mu_e, device); 
	return NuSolutionTensors::Eps2Cartesian(PhysicsFloats::ToTensor(mW, device), PhysicsFloats::ToTensor(mNu, device), _mu); 
}

torch::Tensor NuSolutionFloats::wCartesian(float b_px, float b_py, float b_pz, float b_e, 
			float mu_px, float mu_py, float mu_pz, float mu_e, int factor, std::string device)
{
	torch::Tensor _mu = PhysicsFloats::ToTensor(mu_px, mu_py, mu_pz, mu_e, device);
	torch::Tensor _b = PhysicsFloats::ToTensor(b_px, b_py, b_pz, b_e, device);
	return NuSolutionTensors::wCartesian(_b, _mu, factor); 
}

torch::Tensor NuSolutionFloats::wPolar(float b_pt, float b_eta, float b_phi, float b_e, 
			float mu_pt, float mu_eta, float mu_phi, float mu_e, int factor, std::string device)
{
	torch::Tensor _mu = PhysicsFloats::ToPxPyPzE(b_pt, b_eta, b_phi, b_e, device); 
	torch::Tensor _b = PhysicsFloats::ToPxPyPzE(mu_pt, mu_eta, mu_phi, mu_e, device); 
	return NuSolutionTensors::wCartesian(_b, _mu, factor);
}

torch::Tensor NuSolutionFloats::Omega2Cartesian(float b_px, float b_py, float b_pz, float b_e, 
			float mu_px, float mu_py, float mu_pz, float mu_e, std::string device)
{
	torch::Tensor _mu = PhysicsFloats::ToTensor(mu_px, mu_py, mu_pz, mu_e, device);
	torch::Tensor _b = PhysicsFloats::ToTensor(b_px, b_py, b_pz, b_e, device);

	return NuSolutionTensors::Omega2Cartesian(_b, _mu); 
}

torch::Tensor NuSolutionFloats::Omega2Polar(float b_pt, float b_eta, float b_phi, float b_e, 
			float mu_pt, float mu_eta, float mu_phi, float mu_e, std::string device)
{
	torch::Tensor _mu = PhysicsFloats::ToPxPyPzE(b_pt, b_eta, b_phi, b_e, device); 
	torch::Tensor _b = PhysicsFloats::ToPxPyPzE(mu_pt, mu_eta, mu_phi, mu_e, device); 
	return NuSolutionTensors::Omega2Cartesian(_b, _mu);
}
