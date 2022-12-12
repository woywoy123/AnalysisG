#ifndef H_NUSOLFLOAT
#define H_NUSOLFLOAT

#include <torch/extension.h>
#include <iostream>

namespace NuSolutionFloat
{
	torch::Tensor x0p(float bPt, float bEta, float bPhi, float bE, float MassTop, float MassW); 



}
#endif
