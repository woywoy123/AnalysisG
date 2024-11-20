#ifndef TRANSFORM_PYCU_H
#define TRANSFORM_PYCU_H

#include <torch/torch.h>

namespace transform_ {
    torch::Tensor PxPyPz(torch::Tensor* pmu); 
    torch::Tensor PxPyPzE(torch::Tensor* pmu); 
    torch::Tensor Px(torch::Tensor* pt, torch::Tensor* phi); 
    torch::Tensor Py(torch::Tensor* pt, torch::Tensor* phi); 
    torch::Tensor Pz(torch::Tensor* pt, torch::Tensor* eta);
    torch::Tensor PxPyPz(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi); 
    torch::Tensor PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* e); 

    torch::Tensor PtEtaPhi(torch::Tensor* pmc);
    torch::Tensor PtEtaPhiE(torch::Tensor* pmc);
    torch::Tensor Pt(torch::Tensor* px, torch::Tensor* py);
    torch::Tensor Phi(torch::Tensor* px, torch::Tensor* py);
    torch::Tensor Eta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);
    torch::Tensor PtEtaPhi(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz);
    torch::Tensor PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e);
}

#endif
