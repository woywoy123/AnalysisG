#ifndef H_NUSOL_FLOATS
#define H_NUSOL_FLOATS

#include <torch/extension.h>
#include <iostream>

namespace NuSolutionFloats
{
	torch::Tensor x0Polar(float LPt, float LEta, float LPhi, float LE, float MassH, float MassL, std::string device); 
	torch::Tensor x0Cartesian(float LPx, float LPy, float LPz, float LE, float MassH, float MassL, std::string device); 
}
#endif
