#include <torch/extension.h>
#include "cartesian-cuda/cartesian.h"
#include "polar-cuda/polar.h"

TORCH_LIBRARY(TransformCuda, m)
{
    m.def("Px",      &Transform::CUDA::Px); 
    m.def("Py",      &Transform::CUDA::Py); 
    m.def("Pz",      &Transform::CUDA::Pz); 
    m.def("PxPyPz",  &Transform::CUDA::PxPyPz); 
    m.def("PxPyPzE", &Transform::CUDA::PxPyPzE); 

    m.def("Pt",        &Transform::CUDA::Pt); 
    m.def("Eta",       &Transform::CUDA::Eta); 
    m.def("Phi",       &Transform::CUDA::Phi); 
    m.def("PtEtaPhi",  &Transform::CUDA::PtEtaPhi); 
    m.def("PtEtaPhiE", &Transform::CUDA::PtEtaPhiE); 
}
