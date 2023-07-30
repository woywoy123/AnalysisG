#ifndef H_NUSOL_CUDA_POLAR
#define H_NUSOL_CUDA_POLAR

#include <nusol/nusol-cuda/nusol.h>
#include <tranform/cartesian-cuda/cartesian.h>

namespace NuSol  
{
    namespace CUDA 
    {
        namespace Polar
        {
            const std::map<std::string, torch::Tensor> Nu(
                    torch::Tensor pmu_b, torch::Tensor pmu_mu, 
                    torch::Tensor met_phi, torch::Tensor masses, 
                    torch::Tensor sigma, const double null)
            {
                torch::Tensor pmc_b  = Transform::CUDA::PxPyPzE(pmu_b);  
                torch::Tensor pmc_mu = Transform::CUDA::PxPyPzE(pmu_mu); 
                    
                
            }; 
        }
    }
}
#endif
