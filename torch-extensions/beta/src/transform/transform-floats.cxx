#include <torch/extension.h>
#include <transform/cartesian-floats/cartesian.h>
#include <transform/polar-floats/polar.h>

TORCH_LIBRARY(TransformFloats, m)
{
    m.def("Px", &Transform::Floats::Px); 
    m.def("Py", &Transform::Floats::Py); 
    m.def("Pz", &Transform::Floats::Pz); 
    m.def("PxPyPz", &Transform::Floats::PxPyPz); 

    m.def("Pt", &Transform::Floats::Pt); 
    m.def("Eta", &Transform::Floats::Eta); 
    m.def("Phi", &Transform::Floats::Phi); 
    m.def("PtEtaPhi", &Transform::Floats::PtEtaPhi);
}
