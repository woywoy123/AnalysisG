#include "../Headers/Tensors.h"

torch::Tensor OperatorsTensors::Dot(torch::Tensor v1, torch::Tensor v2)
{
	return (v1*v2).sum({-1}, true); 
}

torch::Tensor OperatorsTensors::CosTheta(torch::Tensor v1, torch::Tensor v2)
{
	torch::Tensor v1_2 = Dot(v1, v1);
	torch::Tensor v2_2 = Dot(v2, v2);
	torch::Tensor dot = Dot(v1, v2); 
	return dot/( torch::sqrt(v1_2 * v2_2) ); 
}

torch::Tensor OperatorsTensors::_SinTheta(torch::Tensor cos)
{
	return torch::sqrt( 1 - cos.pow(2)); 
}


torch::Tensor OperatorsTensors::SinTheta(torch::Tensor v1, torch::Tensor v2)
{
	return _SinTheta(CosTheta(v1, v2)); 
}
