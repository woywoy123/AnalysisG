#ifndef H_PHYSICS_CUDA_POLAR
#define H_PHYSICS_CUDA_POLAR

#include <physics/physics-cuda/physics.h>
#include <transform/cartesian-cuda/cartesian.h>

namespace Physics
{
    namespace CUDA
    {
        namespace Polar
        {
            const torch::Tensor P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi)
            {
                torch::Tensor Pmc = Transform::CUDA::PxPyPz(pt, eta, phi); 
                return Physics::CUDA::P2(Pmc); 
            } 
            const torch::Tensor P2(torch::Tensor Pmu)
            {
                torch::Tensor Pmc = Transform::CUDA::PxPyPzE(Pmu); 
                return Physics::CUDA::P2(Pmc);
            }

            const torch::Tensor P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi)
            {
                 torch::Tensor Pmc = Transform::CUDA::PxPyPz(pt, eta, phi); 
                 return Physics::CUDA::P(Pmc); 
            }

            const torch::Tensor P(torch::Tensor Pmu)
            {
                torch::Tensor pmc = Transform::CUDA::PxPyPzE(Pmu); 
                return Physics::CUDA::P(pmc);
            }

            const torch::Tensor Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e)
            { 
                torch::Tensor pmc = Transform::CUDA::PxPyPzE(pt, eta, phi, e); 
                return Physics::CUDA::Beta2(pmc); 
            }

            const torch::Tensor Beta2(torch::Tensor pmu)
            {
                torch::Tensor pmc = Transform::CUDA::PxPyPzE(pmu); 
                return Physics::CUDA::Beta2(pmc); 
            }

            const torch::Tensor Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e)
            { 
                torch::Tensor pmc = Transform::CUDA::PxPyPzE(pt, eta, phi, e); 
                return Physics::CUDA::Beta(pmc); 
            }

            const torch::Tensor Beta(torch::Tensor pmu)
            {
                torch::Tensor pmc = Transform::CUDA::PxPyPzE(pmu); 
                return Physics::CUDA::Beta(pmc); 
            }
        }
    }
}

#endif
