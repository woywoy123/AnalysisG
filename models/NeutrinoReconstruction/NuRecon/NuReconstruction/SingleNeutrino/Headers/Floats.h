#ifndef H_SINGLENU_FLOATS
#define H_SINGLENU_FLOATS

#include <torch/extension.h>
#include <iostream>
#include "../../Physics/Floats/Headers/PhysicsFloats.h"

namespace SingleNu
{


	namespace Tensors
	{
		torch::Tensor Sigma2(torch::Tensor sxx, torch::Tensor sxy, torch::Tensor syx, torch::Tensor syy);
		torch::Tensor V0(torch::Tensor metx, torch::Tensor mety); 
	}

	namespace Floats
	{
		torch::Tensor Sigma2(double sxx, double sxy, double syx, double syy, std::string device); 
		static torch::Tensor V0(double metx, double mety, std::string device)
		{
			return SingleNu::Tensors::V0(PhysicsFloats::ToTensor(metx, device), 
					PhysicsFloats::ToTensor(mety, device)); 
		}
	}
}


#endif 
