#ifndef H_TRANSFORM_TENSORS_CARTESIAN
#define H_TRANSFORM_TENSORS_CARTESIAN
#include <torch/torch.h>

namespace Transform
{
    namespace Tensors
    {
	    torch::Tensor Px(torch::Tensor pt, torch::Tensor phi); 
	    torch::Tensor Py(torch::Tensor pt, torch::Tensor phi); 
	    torch::Tensor Pz(torch::Tensor pt, torch::Tensor eta); 
	    torch::Tensor PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);
        torch::Tensor PxPyPzE(torch::Tensor Pmu);

    }
}

#endif
