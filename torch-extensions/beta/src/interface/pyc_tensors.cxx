#include <torch/extension.h>
#include "pyc_tensors.h"

torch::Tensor pyc::transform::separate::Pt(torch::Tensor px, torch::Tensor py){ return Transform::Tensors::Pt(px, py); }
torch::Tensor pyc::transform::separate::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ return Transform::Tensors::Eta(px, py, pz); }
torch::Tensor pyc::transform::separate::Phi(torch::Tensor px, torch::Tensor py){ return Transform::Tensors::Phi(px, py); }
torch::Tensor pyc::transform::separate::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ return Transform::Tensors::PtEtaPhi(px, py, pz); }
torch::Tensor pyc::transform::separate::PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ return Transform::Tensors::PtEtaPhiE(px, py, pz, e); }

torch::Tensor pyc::transform::separate::Px(torch::Tensor pt, torch::Tensor phi){ return Transform::Tensors::Px(pt, phi); }
torch::Tensor pyc::transform::separate::Py(torch::Tensor pt, torch::Tensor phi){ return Transform::Tensors::Py(pt, phi); }
torch::Tensor pyc::transform::separate::Pz(torch::Tensor pt, torch::Tensor eta){ return Transform::Tensors::Pz(pt, eta); }
torch::Tensor pyc::transform::separate::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){ return Transform::Tensors::PxPyPz(pt, eta, phi); }
torch::Tensor pyc::transform::separate::PxPyPzE(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ return Transform::Tensors::PxPyPzE(pt, eta, phi, e); }

torch::Tensor pyc::transform::combined::Pt(torch::Tensor pmc){ return Transform::Tensors::Pt(pmc); }
torch::Tensor pyc::transform::combined::Eta(torch::Tensor pmc){ return Transform::Tensors::Eta(pmc); }
torch::Tensor pyc::transform::combined::Phi(torch::Tensor pmc){ return Transform::Tensors::Phi(pmc); }
torch::Tensor pyc::transform::combined::PtEtaPhi(torch::Tensor pmc){ return Transform::Tensors::PtEtaPhi(pmc); }
torch::Tensor pyc::transform::combined::PtEtaPhiE(torch::Tensor pmc){ return Transform::Tensors::PtEtaPhiE(pmc); }

torch::Tensor pyc::transform::combined::Px(torch::Tensor pmu){ return Transform::Tensors::Px(pmu); }
torch::Tensor pyc::transform::combined::Py(torch::Tensor pmu){ return Transform::Tensors::Py(pmu); }
torch::Tensor pyc::transform::combined::Pz(torch::Tensor pmu){ return Transform::Tensors::Pz(pmu); }
torch::Tensor pyc::transform::combined::PxPyPz(torch::Tensor pmu){ return Transform::Tensors::PxPyPz(pmu); }
torch::Tensor pyc::transform::combined::PxPyPzE(torch::Tensor pmu){ return Transform::Tensors::PxPyPzE(pmu); }

TORCH_LIBRARY(pyc_tensor, m)
{
    // transformation classes for Tensors
    m.def("transform_separate_Px",        &pyc::transform::separate::Px);
    m.def("transform_separate_Py",        &pyc::transform::separate::Py);
    m.def("transform_separate_Pz",        &pyc::transform::separate::Pz);
    m.def("transform_separate_PxPyPz",    &pyc::transform::separate::PxPyPz);
    m.def("transform_separate_PxPyPzE",   &pyc::transform::separate::PxPyPzE);

    m.def("transform_combined_Px",        &pyc::transform::combined::Px);
    m.def("transform_combined_Py",        &pyc::transform::combined::Py);
    m.def("transform_combined_Pz",        &pyc::transform::combined::Pz);
    m.def("transform_combined_PxPyPz",    &pyc::transform::combined::PxPyPz);
    m.def("transform_combined_PxPyPzE",   &pyc::transform::combined::PxPyPzE);

    m.def("transform_separate_Pt",        &pyc::transform::separate::Pt);
    m.def("transform_separate_Phi",       &pyc::transform::separate::Phi);
    m.def("transform_separate_Eta",       &pyc::transform::separate::Eta);
    m.def("transform_separate_PtEtaPhi",  &pyc::transform::separate::PtEtaPhi);
    m.def("transform_separate_PtEtaPhiE", &pyc::transform::separate::PtEtaPhiE);

    m.def("transform_combined_Pt",        &pyc::transform::combined::Pt);
    m.def("transform_combined_Phi",       &pyc::transform::combined::Phi);
    m.def("transform_combined_Eta",       &pyc::transform::combined::Eta);
    m.def("transform_combined_PtEtaPhi",  &pyc::transform::combined::PtEtaPhi);
    m.def("transform_combined_PtEtaPhiE", &pyc::transform::combined::PtEtaPhiE);
}
