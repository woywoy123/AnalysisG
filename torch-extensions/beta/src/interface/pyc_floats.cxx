#include <torch/extension.h>
#include "pyc_floats.h"

double pyc::transform::separate::Pt(double px, double py){ return Transform::Floats::Pt(px, py); }
double pyc::transform::separate::Eta(double px, double py, double pz){ return Transform::Floats::Eta(px, py, pz); }
double pyc::transform::separate::Phi(double px, double py){ return Transform::Floats::Phi(px, py); }
std::vector<double> pyc::transform::separate::PtEtaPhi(double px, double py, double pz){ return Transform::Floats::PtEtaPhi(px, py, pz); }
std::vector<double> pyc::transform::separate::PtEtaPhiE(double px, double py, double pz, double e){ return Transform::Floats::PtEtaPhiE(px, py, pz, e); }

std::vector<double> pyc::transform::combined::Pt(std::vector<std::vector<double>> pmc){ return Transform::Floats::Pt(pmc); }
std::vector<double> pyc::transform::combined::Eta(std::vector<std::vector<double>> pmc){ return Transform::Floats::Eta(pmc); }
std::vector<double> pyc::transform::combined::Phi(std::vector<std::vector<double>> pmc){ return Transform::Floats::Phi(pmc); }
std::vector<std::vector<double>> pyc::transform::combined::PtEtaPhi(std::vector<std::vector<double>> pmc){ return Transform::Floats::PtEtaPhi(pmc); }
std::vector<std::vector<double>> pyc::transform::combined::PtEtaPhiE(std::vector<std::vector<double>> pmc){ return Transform::Floats::PtEtaPhiE(pmc); }

double pyc::transform::separate::Px(double pt, double phi){ return Transform::Floats::Px(pt, phi); }
double pyc::transform::separate::Py(double pt, double phi){ return Transform::Floats::Py(pt, phi); }
double pyc::transform::separate::Pz(double pt, double eta){ return Transform::Floats::Pz(pt, eta); }
std::vector<double> pyc::transform::separate::PxPyPz(double pt, double eta, double phi){ return Transform::Floats::PxPyPz(pt, eta, phi); }
std::vector<double> pyc::transform::separate::PxPyPzE(double pt, double eta, double phi, double e){ return Transform::Floats::PxPyPzE(pt, eta, phi, e); }

std::vector<double> pyc::transform::combined::Px(std::vector<std::vector<double>> pmu){ return Transform::Floats::Px(pmu); }
std::vector<double> pyc::transform::combined::Py(std::vector<std::vector<double>> pmu){ return Transform::Floats::Py(pmu); }
std::vector<double> pyc::transform::combined::Pz(std::vector<std::vector<double>> pmu){ return Transform::Floats::Pz(pmu); }
std::vector<std::vector<double>> pyc::transform::combined::PxPyPz(std::vector<std::vector<double>> pmu){ return Transform::Floats::PxPyPz(pmu); }
std::vector<std::vector<double>> pyc::transform::combined::PxPyPzE(std::vector<std::vector<double>> pmu){ return Transform::Floats::PxPyPzE(pmu); }

TORCH_LIBRARY(pyc_float, m)
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
