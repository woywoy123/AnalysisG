#ifndef H_PHYSICS_CUDA_CARTESIAN
#define H_PHYSICS_CUDA_CARTESIAN

#include <physics/physics-cuda/physics.h>

namespace Physics
{
    namespace CUDA 
    {
        namespace Cartesian 
        {
            const torch::Tensor P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
            { 
                return Physics::CUDA::P2(px, py, pz); 
            }

            const torch::Tensor P2(torch::Tensor pmc)
            {
                return Physics::CUDA::P2(pmc); 
            }

            const torch::Tensor P(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
            { 
                return Physics::CUDA::P(px, py, pz); 
            }

            const torch::Tensor P(torch::Tensor pmc)
            {
                return Physics::CUDA::P(pmc); 
            }

            const torch::Tensor Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e)
            { 
                return Physics::CUDA::Beta2(px, py, pz, e); 
            }

            const torch::Tensor Beta2(torch::Tensor pmc)
            {
                return Physics::CUDA::Beta2(pmc); 
            }

            const torch::Tensor Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e)
            { 
                return Physics::CUDA::Beta(px, py, pz, e); 
            }

            const torch::Tensor Beta(torch::Tensor pmc)
            {
                return Physics::CUDA::Beta(pmc); 
            }
        }
    }
}

#endif
