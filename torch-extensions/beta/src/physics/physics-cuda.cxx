#include <torch/extension.h>
#include <physics/physics-cuda/polar.h>
#include <physics/physics-cuda/cartesian.h>

TORCH_LIBRARY(PhysicsPolar, m)
{
    m.def("PolarP2", &Physics::CUDA::Polar::P2); 
    m.def("PolarP", &Physics::CUDA::Polar::P);

    m.def("CartesianP2", &Physics::CUDA::Cartesian::P2); 
    m.def("CartesianP", &Physics::CUDA::Cartesian::P); 
}
