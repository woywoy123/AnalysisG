#ifndef H_NUSOL_TENSORS
#define H_NUSOL_TENSORS

#include <torch/extension.h>
#include <iostream>

namespace NuSolutionTensors
{
	torch::Tensor x0(torch::Tensor PolarL, torch::Tensor MassH, torch::Tensor MassL); 

}


#endif 
