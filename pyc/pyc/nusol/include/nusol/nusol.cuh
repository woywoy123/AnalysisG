#ifndef CUNUSOL_CUDA_H
#define CUNUSOL_CUDA_H

#include <torch/torch.h>
#include <string>
#include <map>


namespace nusol_ {
    std::map<std::string, torch::Tensor> BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses); 
    std::map<std::string, torch::Tensor> Intersection(torch::Tensor* A, torch::Tensor* B, double nulls); 

    std::map<std::string, torch::Tensor> Nu(
            torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
            torch::Tensor* masses, torch::Tensor* sigma , double null); 

    std::map<std::string, torch::Tensor> NuNu(
            torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy, torch::Tensor* masses, double null); 

}

#endif
