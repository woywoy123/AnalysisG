#ifndef CUNUSOL_CUDA_H
#define CUNUSOL_CUDA_H

#include <map>
#include <string>
#include <torch/torch.h>

namespace nusol_ {
    std::map<std::string, torch::Tensor> BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses); 
    std::map<std::string, torch::Tensor> BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, double mT, double mW, double mN); 
    std::map<std::string, torch::Tensor> Intersection(torch::Tensor* A, torch::Tensor* B, double nulls); 

    std::map<std::string, torch::Tensor> Nu(
            torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
            torch::Tensor* masses, torch::Tensor* sigma , double null
    ); 

    std::map<std::string, torch::Tensor> NuNu(
            torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy, double null, torch::Tensor* m1, torch::Tensor* m2,
            const double step, const double tolerance, const unsigned int timeout
    ); 

    std::map<std::string, torch::Tensor> combinatorial(
               torch::Tensor* edge_index, torch::Tensor* batch , torch::Tensor* pmc, 
               torch::Tensor* pid       , torch::Tensor* met_xy, 
               double mT  = 172.62*1000 , double mW = 80.385*1000, double null = 1e-10, double perturb = 1e-3, 
               long steps = 100, bool gev = false
    ); 
}

#endif

