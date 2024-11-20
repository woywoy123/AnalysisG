#ifndef CUNUSOL_CUDA_H
#define CUNUSOL_CUDA_H

#include <torch/torch.h>

namespace nusol_ {
    torch::Tensor BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses); 

}

#endif
