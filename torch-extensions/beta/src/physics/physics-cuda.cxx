#include <torch/extension.h>
#include <physics/physics-cuda/polar.h>

TORCH_LIBRARY(PhysicsPolar, m)
{
    m.def("P2", &Physics::CUDA::Polar::P2); 
}
