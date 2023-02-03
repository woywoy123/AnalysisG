#ifndef H_OPERATORS_TENSORS
#define H_OPERATORS_TENSORS
#include <torch/extension.h>

namespace OperatorsTensors
{
	torch::Tensor Dot(torch::Tensor v1, torch::Tensor v2); 
	torch::Tensor CosTheta(torch::Tensor v1, torch::Tensor v2);
	torch::Tensor SinTheta(torch::Tensor v1, torch::Tensor v2); 
	torch::Tensor _SinTheta(torch::Tensor cos); 
}
#endif 
