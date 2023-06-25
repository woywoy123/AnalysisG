#ifndef H_PHYSICS_CUDA
#define H_PHYSICS_CUDA
#include <torch/torch.h>
torch::Tensor _P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
torch::Tensor _P2(torch::Tensor pmc); 

namespace Physics
{
    namespace CUDA
    {
        const torch::Tensor P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ return _P2(px, py, pz); }
        const torch::Tensor P2(torch::Tensor pmc){ return _P2(pmc); }
    }
}
#endif
