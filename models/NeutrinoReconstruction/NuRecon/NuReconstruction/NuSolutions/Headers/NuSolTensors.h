#ifndef H_NUSOL_TENSORS
#define H_NUSOL_TENSORS

#include <torch/extension.h>
#include <iostream>

namespace NuSolutionTensors
{
	torch::Tensor x0Polar(torch::Tensor PolarL, torch::Tensor MassH, torch::Tensor MassL); 
	torch::Tensor x0Cartesian(torch::Tensor CartesianL, torch::Tensor MassH, torch::Tensor MassL);
}


#endif 
