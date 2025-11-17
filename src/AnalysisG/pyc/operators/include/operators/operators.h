#ifndef OPERATORS_H
#define OPERATORS_H

#include <torch/torch.h>

namespace operators_ {
    torch::Tensor Dot(torch::Tensor* v1, torch::Tensor* v2); 
    torch::Tensor Cross(torch::Tensor* v1, torch::Tensor* v2); 
    torch::Tensor Determinant(torch::Tensor* matrix);

    torch::Tensor CosTheta(torch::Tensor* v1, torch::Tensor* v2);
    torch::Tensor SinTheta(torch::Tensor* v1, torch::Tensor* v2); 
    torch::Tensor Pi_2(torch::Tensor* trn); 

    torch::Tensor Rx(torch::Tensor* angle); 
    torch::Tensor Ry(torch::Tensor* angle); 
    torch::Tensor Rz(torch::Tensor* angle); 
    torch::Tensor RT(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* base); 
    torch::Tensor CoFactors(torch::Tensor* matrix);

    std::tuple<torch::Tensor, torch::Tensor> Inverse(torch::Tensor* matrix);
    std::tuple<torch::Tensor, torch::Tensor> Eigenvalue(torch::Tensor* matrix);
}

#endif
