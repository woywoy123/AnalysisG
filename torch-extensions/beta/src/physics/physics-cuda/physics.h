#ifndef H_PHYSICS_CUDA
#define H_PHYSICS_CUDA
#include <torch/torch.h>

torch::Tensor _P2(torch::Tensor pmc);
torch::Tensor _P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
torch::Tensor _P(torch::Tensor pmc); 
torch::Tensor _P(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
torch::Tensor _Beta2(torch::Tensor pmc); 
torch::Tensor _Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
torch::Tensor _Beta(torch::Tensor pmc); 
torch::Tensor _Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 

namespace Physics
{
    namespace CUDA
    {
        const torch::Tensor P2(torch::Tensor pmc){ return _P2(pmc); }
        const torch::Tensor P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ return _P2(px, py, pz); }

        const torch::Tensor P(torch::Tensor pmc){ return _P(pmc); }
        const torch::Tensor P(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ return _P(px, py, pz); }

        const torch::Tensor Beta2(torch::Tensor pmc){ return _Beta2(pmc); }
        const torch::Tensor Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ return _Beta2(px, py, pz, e); }
        
        const torch::Tensor Beta(torch::Tensor pmc){ return _Beta(pmc); }
        const torch::Tensor Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ return _Beta(px, py, pz, e); }
    }
}
#endif
