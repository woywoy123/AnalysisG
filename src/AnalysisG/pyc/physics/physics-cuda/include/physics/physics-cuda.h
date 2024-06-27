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

torch::Tensor _M2(torch::Tensor pmc); 
torch::Tensor _M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
torch::Tensor _M(torch::Tensor pmc); 
torch::Tensor _M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 

torch::Tensor _Mt2(torch::Tensor pmc); 
torch::Tensor _Mt2(torch::Tensor pz, torch::Tensor e); 
torch::Tensor _Mt(torch::Tensor pmc); 
torch::Tensor _Mt(torch::Tensor pz, torch::Tensor e); 

torch::Tensor _Theta(torch::Tensor pmc); 
torch::Tensor _Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 

torch::Tensor _DeltaR(torch::Tensor pmu1, torch::Tensor pmu2); 
torch::Tensor _DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2); 


namespace physics {
    namespace cuda {
        torch::Tensor P2(torch::Tensor pmc); 
        torch::Tensor P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
        torch::Tensor P(torch::Tensor pmc); 
        torch::Tensor P(torch::Tensor px, torch::Tensor py, torch::Tensor pz);
        torch::Tensor Beta2(torch::Tensor pmc); 
        torch::Tensor Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
        torch::Tensor Beta(torch::Tensor pmc); 
        torch::Tensor Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
        torch::Tensor M2(torch::Tensor pmc); 
        torch::Tensor M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
        torch::Tensor M(torch::Tensor pmc); 
        torch::Tensor M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
        torch::Tensor Mt2(torch::Tensor pmc); 
        torch::Tensor Mt2(torch::Tensor pz, torch::Tensor e); 
        torch::Tensor Mt(torch::Tensor pmc); 
        torch::Tensor Mt(torch::Tensor pz, torch::Tensor e);
        torch::Tensor Theta(torch::Tensor pmc); 
        torch::Tensor Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz);
        torch::Tensor DeltaR(torch::Tensor pmu1, torch::Tensor pmu2); 
        torch::Tensor DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2); 
    }
}
#endif
