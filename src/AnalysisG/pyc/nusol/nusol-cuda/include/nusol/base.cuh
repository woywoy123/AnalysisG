#ifndef NUSOL_CUDA_BASE_H
#define NUSOL_CUDA_BASE_H

#include <vector>
#include <torch/torch.h>

torch::Tensor ShapeMatrix(torch::Tensor* inpt, std::vector<long> vec); 
torch::Tensor ExpandMatrix(torch::Tensor* inpt, torch::Tensor* source); 
torch::Tensor BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* mWtnu);

#endif
