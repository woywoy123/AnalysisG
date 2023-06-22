#include <torch/extension.h>
#include "cartesian-cuda/cartesian.h"

TORCH_LIBRARY(TransformCuda, m)
{
    m.def("Px",      &Transform::CUDA::Px); 
    m.def("Py",      &Transform::CUDA::Py); 
    m.def("Pz",      &Transform::CUDA::Pz); 
    m.def("PxPyPz",  &Transform::CUDA::PxPyPz); 
    m.def("PxPyPzE", &Transform::CUDA::PxPyPzE); 
}
