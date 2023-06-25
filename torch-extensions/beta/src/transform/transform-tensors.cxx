#include <torch/extension.h>
#include "cartesian-tensors/cartesian.h"
#include "polar-tensors/polar.h"

TORCH_LIBRARY(TransformTensors, m)
{
    m.def("Px",      &Transform::Tensors::Px); 
    m.def("Py",      &Transform::Tensors::Py); 
    m.def("Pz",      &Transform::Tensors::Pz); 
    m.def("PxPyPz",  &Transform::Tensors::PxPyPz); 
    m.def("PxPyPzE", &Transform::Tensors::PxPyPzE); 

    m.def("Pt",        &Transform::Tensors::Pt); 
    m.def("Eta",       &Transform::Tensors::Eta); 
    m.def("Phi",       &Transform::Tensors::Phi); 
    m.def("PtEtaPhi",  &Transform::Tensors::PtEtaPhi); 
    m.def("PtEtaPhiE", &Transform::Tensors::PtEtaPhiE); 
}
