#include <transform/cartesian-cuda/cartesian.h>
#include "physics.h"

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
                torch::Tensor Pmc = Transform::CUDA::PxPyPzE(Pmu); 
                return Physics::CUDA::P(Pmc);
            }
        }
    }
}
