#include "physics.h"

namespace Physics
{
    namespace CUDA 
    {
        namespace Cartesian 
        {
            const torch::Tensor P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ return Physics::CUDA::P2(px, py, pz); }
            const torch::Tensor P(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ return Physics::CUDA::P(px, py, pz); }
        }
    }
}
