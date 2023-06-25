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
                return Physics::CUDA::P2(pt, eta, phi); 
            } 
        }


    }


}
