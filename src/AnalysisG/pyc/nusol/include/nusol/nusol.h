#ifndef NUSOL_H
#define NUSOL_H

#include <map>
#include <tuple>
#include <string>
#include <torch/torch.h>

namespace nusol_ {
    torch::Tensor BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses); 

    std::tuple<torch::Tensor, torch::Tensor> Intersection(torch::Tensor* A, torch::Tensor* B, double nulls); 

    std::map<std::string, torch::Tensor> Nu(
            torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
            torch::Tensor* masses, torch::Tensor* sigma , double null); 

    std::map<std::string, torch::Tensor> NuNu(
            torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy, double null, torch::Tensor* m1, torch::Tensor* m2 = nullptr); 
}

#endif

