#ifndef H_TRANSFORM_CUDA_CARTESIAN
#define H_TRANSFORM_CUDA_CARTESIAN

#include <torch/torch.h>
torch::Tensor _Px(torch::Tensor pt, torch::Tensor phi); 
torch::Tensor _Py(torch::Tensor pt, torch::Tensor phi); 
torch::Tensor _Pz(torch::Tensor pt, torch::Tensor eta); 
torch::Tensor _PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
torch::Tensor _PxPyPzE(torch::Tensor Pmu); 

namespace Transform
{
    namespace CUDA
    {
        const torch::Tensor Px(torch::Tensor pt, torch::Tensor phi){return _Px(pt, phi);} 
        const torch::Tensor Py(torch::Tensor pt, torch::Tensor phi){return _Py(pt, phi);} 
        const torch::Tensor Pz(torch::Tensor pt, torch::Tensor eta){return _Pz(pt, eta);} 
        const torch::Tensor PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){return _PxPyPz(pt, eta, phi);} 
        const torch::Tensor PxPyPzE(torch::Tensor Pmu){return _PxPyPzE(Pmu);} 
    }
}

#endif
