#ifndef H_NUSOL_FLOATS
#define H_NUSOL_FLOATS

#include <torch/extension.h>
#include <iostream>

namespace NuSolutionFloats
{
	torch::Tensor x0Polar(float LPt, float LEta, float LPhi, float LE, float MassH, float MassL, std::string device); 
	torch::Tensor x0Cartesian(float LPx, float LPy, float LPz, float LE, float MassH, float MassL, std::string device); 
	
	torch::Tensor SxPolar(float bPt, float bEta, float bPhi, float bE, 
			float muPt, float muEta, float muPhi, float muE, 
			float mW, float mNu, std::string device);

	torch::Tensor SyPolar(float bPt, float bEta, float bPhi, float bE, 
			float muPt, float muEta, float muPhi, float muE, 
			float mTop, float mW, float mNu, std::string device);

	torch::Tensor Eps2Cartesian(float mu_px, float mu_py, float mu_pz, float mu_e, 
			float mW, float mNu,std::string device);
	torch::Tensor Eps2Polar(float mu_pt, float mu_eta, float mu_phi, float mu_e, 
			float mW, float mNu, std::string device);

	torch::Tensor wCartesian(float b_px, float b_py, float b_pz, float b_e, 
			float mu_px, float mu_py, float mu_pz, float mu_e, int factor, std::string device); 
	torch::Tensor wPolar(float b_pt, float b_eta, float b_phi, float b_e, 
			float mu_pt, float mu_eta, float mu_phi, float mu_e, int factor, std::string device);

	torch::Tensor Omega2Cartesian(float b_px, float b_py, float b_pz, float b_e, 
			float mu_px, float mu_py, float mu_pz, float mu_e, std::string device);
	torch::Tensor Omega2Polar(float b_pt, float b_eta, float b_phi, float b_e, 
			float mu_pt, float mu_eta, float mu_phi, float mu_e, std::string device);

}
#endif
