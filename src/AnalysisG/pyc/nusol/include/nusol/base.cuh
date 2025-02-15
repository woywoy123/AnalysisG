#ifndef CU_NUSOL_BASE_H
#define CU_NUSOL_BASE_H

#include <map>
#include <string>
#include <torch/torch.h>
#include <atomic/cuatomic.cuh>

namespace nusol_ {
    std::map<std::string, torch::Tensor> BaseDebug(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses); 
    std::map<std::string, torch::Tensor> BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses); 
    std::map<std::string, torch::Tensor> BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, double mT, double mW, double mN); 
    std::map<std::string, torch::Tensor> Intersection(torch::Tensor* A, torch::Tensor* B, double nulls); 

    std::map<std::string, torch::Tensor> Nu(torch::Tensor* H, torch::Tensor* sigma, torch::Tensor* met_xy, double null); 
    std::map<std::string, torch::Tensor> NuNu(
            torch::Tensor* H1_, torch::Tensor* H1_perp, torch::Tensor* H2_, torch::Tensor* H2_perp, torch::Tensor* met_xy, double null
    ); 

    std::map<std::string, torch::Tensor> Nu(
            torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, torch::Tensor* masses, torch::Tensor* sigma , double null
    ); 

    std::map<std::string, torch::Tensor> Nu(
            torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, torch::Tensor* sigma , double null, double massT, double massW
    ); 

    std::map<std::string, torch::Tensor> NuNu(
            torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy, double null, torch::Tensor* m1, torch::Tensor* m2
    ); 

    std::map<std::string, torch::Tensor> NuNu(
            torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy, double null, double massT1, double massW1, double massT2, double massW2
    ); 

}
#endif
