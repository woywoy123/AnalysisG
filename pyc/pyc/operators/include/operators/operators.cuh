#ifndef CU_OPERATORS_H
#define CU_OPERATORS_H

#include <torch/torch.h>

namespace operators_ {
    torch::Tensor Dot(torch::Tensor* v1, torch::Tensor* v2); 
    torch::Tensor Cross(torch::Tensor* v1, torch::Tensor* v2); 
    torch::Tensor CosTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm = 0);
    torch::Tensor SinTheta(torch::Tensor* v1, torch::Tensor* v2, unsigned int lm = 0); 
    torch::Tensor Rx(torch::Tensor* angle); 
    torch::Tensor Ry(torch::Tensor* angle); 
    torch::Tensor Rz(torch::Tensor* angle); 
    torch::Tensor RT(torch::Tensor* pmu, torch::Tensor* phi, torch::Tensor* theta); 
    torch::Tensor CoFactors(torch::Tensor* matrix);
    torch::Tensor Determinant(torch::Tensor* matrix);
    std::tuple<torch::Tensor, torch::Tensor> Inverse(torch::Tensor* matrix);
    std::tuple<torch::Tensor, torch::Tensor> Eigenvalue(torch::Tensor* matrix); 
}

#endif
