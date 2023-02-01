#ifndef H_VECTOR_TOCARTESIAN_T
#define H_VECTOR_TOCARTESIAN_T

#include <torch/extension.h>

namespace VectorTensors
{
	torch::Tensor Px(torch::Tensor pt, torch::Tensor phi); 
	torch::Tensor Py(torch::Tensor pt, torch::Tensor phi); 
	torch::Tensor Pz(torch::Tensor pt, torch::Tensor eta); 
	torch::Tensor PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);
}

#endif 
