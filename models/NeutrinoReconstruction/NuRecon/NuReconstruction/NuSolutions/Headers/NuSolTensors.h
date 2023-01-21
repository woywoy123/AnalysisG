#ifndef H_NUSOL_TENSORS
#define H_NUSOL_TENSORS

#include <torch/extension.h>
#include <iostream>

namespace NuSolutionTensors
{
	torch::Tensor x0Polar(torch::Tensor PolarL, torch::Tensor MassH, torch::Tensor MassL); 
	torch::Tensor x0Cartesian(torch::Tensor CartesianL, torch::Tensor MassH, torch::Tensor MassL);
	
	torch::Tensor SxCartesian(torch::Tensor _b, torch::Tensor _mu, 
			torch::Tensor mW, torch::Tensor mNu); 
	torch::Tensor SyCartesian(torch::Tensor _b, torch::Tensor _mu, 
			torch::Tensor mTop, torch::Tensor mW, torch::Tensor Sx); 
	torch::Tensor SxSyCartesian(torch::Tensor _b, torch::Tensor _mu, 
			torch::Tensor mTop, torch::Tensor mW, torch::Tensor mNu);

	torch::Tensor Eps2Cartesian(torch::Tensor mW, torch::Tensor mNu, torch::Tensor mu);
	torch::Tensor Eps2Polar(torch::Tensor mW, torch::Tensor mNu, torch::Tensor mu);

	torch::Tensor wCartesian(torch::Tensor _b, torch::Tensor _mu, int factor); 
	torch::Tensor wPolar(torch::Tensor _b, torch::Tensor _mu, int factor); 

	torch::Tensor Omega2Cartesian(torch::Tensor _b, torch::Tensor _mu); 
	torch::Tensor Omega2Polar(torch::Tensor _b, torch::Tensor _mu);

	torch::Tensor AnalyticalSolutionsCartesian(torch::Tensor _b, torch::Tensor _mu, 
			torch::Tensor massTop, torch::Tensor massW, torch::Tensor massNu);
	torch::Tensor AnalyticalSolutionsPolar(torch::Tensor _b, torch::Tensor _mu, 
			torch::Tensor massTop, torch::Tensor massW, torch::Tensor massNu);

	torch::Tensor Rotation(torch::Tensor _b, torch::Tensor _mu);
	torch::Tensor H(torch::Tensor _b, torch::Tensor _mu, 
			torch::Tensor massTop, torch::Tensor massW, torch::Tensor massNu); 




}


#endif 
