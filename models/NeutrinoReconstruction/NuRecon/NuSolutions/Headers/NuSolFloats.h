#ifndef H_NUSOL_FLOAT
#define H_NUSOL_FLOAT

#include <torch/extension.h>
#include <iostream>

namespace NuSolutionFloats
{
	torch::Tensor x0(float LPt, float LEta, float LPhi, float LE, float MassH, float MassL, std::string device); 
}
#endif
