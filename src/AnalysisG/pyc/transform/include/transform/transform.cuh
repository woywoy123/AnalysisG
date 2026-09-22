#ifndef TRANSFORM_CUH
#define TRANSFORM_CUH
#include <torch/torch.h>

namespace transform_ {
    torch::Tensor Px(torch::Tensor* pt, torch::Tensor* phi);
    torch::Tensor Py(torch::Tensor* pt, torch::Tensor* phi);
    torch::Tensor Pz(torch::Tensor* e, torch::Tensor* eta);
    torch::Tensor PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* energy);
    torch::Tensor PxPyPzE(torch::Tensor* pmu);

    torch::Tensor PtEtaPhiE(torch::Tensor* pmc);
    torch::Tensor PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);

    torch::Tensor Pt(torch::Tensor* px, torch::Tensor* py);
    torch::Tensor Phi(torch::Tensor* px, torch::Tensor* py);
    torch::Tensor Eta(torch::Tensor* pz, torch::Tensor* e);

    torch::Tensor Pt(torch::Tensor* pmc);
    torch::Tensor Eta(torch::Tensor* pmc); 
    torch::Tensor Phi(torch::Tensor* pmc);
    torch::Tensor Px(torch::Tensor* pmc);
    torch::Tensor Py(torch::Tensor* pmc);
    torch::Tensor Pz(torch::Tensor* pmc);
}

#endif
