#include <pyc/cupyc.h>

TORCH_LIBRARY(transform_cuda, m){

    m.def("transform_separate_px",        &pyc::transform::separate::Px);
    m.def("transform_separate_py",        &pyc::transform::separate::Py);
    m.def("transform_separate_pz",        &pyc::transform::separate::Pz);
    m.def("transform_separate_pxpypz",    &pyc::transform::separate::PxPyPz);
    m.def("transform_separate_pxpypze",   &pyc::transform::separate::PxPyPzE);

    m.def("transform_combined_px",        &pyc::transform::combined::Px);
    m.def("transform_combined_py",        &pyc::transform::combined::Py);
    m.def("transform_combined_pz",        &pyc::transform::combined::Pz);
    m.def("transform_combined_pxpypz",    &pyc::transform::combined::PxPyPz);
    m.def("transform_combined_pxpypze",   &pyc::transform::combined::PxPyPzE);

    m.def("transform_separate_pt",        &pyc::transform::separate::Pt);
    m.def("transform_separate_phi",       &pyc::transform::separate::Phi);
    m.def("transform_separate_eta",       &pyc::transform::separate::Eta);
    m.def("transform_separate_ptetaphi",  &pyc::transform::separate::PtEtaPhi);
    m.def("transform_separate_ptetaphie", &pyc::transform::separate::PtEtaPhiE);

    m.def("transform_combined_pt",        &pyc::transform::combined::Pt);
    m.def("transform_combined_phi",       &pyc::transform::combined::Phi);
    m.def("transform_combined_eta",       &pyc::transform::combined::Eta);
    m.def("transform_combined_ptetaphi",  &pyc::transform::combined::PtEtaPhi);
    m.def("transform_combined_ptetaphie", &pyc::transform::combined::PtEtaPhiE);
}


