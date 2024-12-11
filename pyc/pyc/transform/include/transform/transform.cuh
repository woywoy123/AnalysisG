#include <torch/torch.h>

#ifndef TRANSFORM_CUH
#define TRANSFORM_CUH

namespace transform_ {
    torch::Tensor Px(torch::Tensor* pt, torch::Tensor* phi);
    torch::Tensor Py(torch::Tensor* pt, torch::Tensor* phi);
    torch::Tensor Pz(torch::Tensor* pt, torch::Tensor* eta);
    torch::Tensor PxPyPz(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi);
    torch::Tensor PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* energy);
    torch::Tensor PxPyPz(torch::Tensor* pmu);
    torch::Tensor PxPyPzE(torch::Tensor* pmu);

    torch::Tensor PtEtaPhi(torch::Tensor* pmc);
    torch::Tensor PtEtaPhi(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

    torch::Tensor PtEtaPhiE(torch::Tensor* pmc);
    torch::Tensor PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

    torch::Tensor Pt(torch::Tensor* px, torch::Tensor* py);
    torch::Tensor Phi(torch::Tensor* px, torch::Tensor* py);

    torch::Tensor Eta(torch::Tensor* pmc); 
    torch::Tensor Eta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);

}

#endif