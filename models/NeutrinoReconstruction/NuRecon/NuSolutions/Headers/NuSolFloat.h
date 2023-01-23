#ifndef H_NUSOLFLOAT
#define H_NUSOLFLOAT

#include <torch/extension.h>
#include <iostream>

namespace NuSolutionFloat
{
	torch::Tensor x0p(double bPt, double bEta, double bPhi, double bE, double MassTop, double MassW); 



}
#endif
