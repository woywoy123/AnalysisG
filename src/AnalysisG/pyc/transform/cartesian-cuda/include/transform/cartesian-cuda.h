#ifndef H_TRANSFORM_CUDA_CARTESIAN
#define H_TRANSFORM_CUDA_CARTESIAN

#include <torch/torch.h>
torch::Tensor _Px(torch::Tensor pt, torch::Tensor phi); 
torch::Tensor _Py(torch::Tensor pt, torch::Tensor phi); 
torch::Tensor _Pz(torch::Tensor pt, torch::Tensor eta); 
torch::Tensor _PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
torch::Tensor _PxPyPzE(torch::Tensor Pmu); 

namespace transform {
    namespace cuda {
        torch::Tensor Cclip(torch::Tensor inpt, int dim); 
        torch::Tensor Px(torch::Tensor pt, torch::Tensor phi);
        torch::Tensor Px(torch::Tensor pmu);
        torch::Tensor Py(torch::Tensor pt, torch::Tensor phi);
        torch::Tensor Py(torch::Tensor pmu);
        torch::Tensor Pz(torch::Tensor pt, torch::Tensor eta);
        torch::Tensor Pz(torch::Tensor pmu);
        torch::Tensor PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi);
        torch::Tensor PxPyPz(torch::Tensor pmu);
        torch::Tensor PxPyPzE(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);
        torch::Tensor PxPyPzE(torch::Tensor Pmu);
    }
}

#endif
