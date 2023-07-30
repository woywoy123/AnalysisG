#ifndef H_NUSOL_TENSOR
#define H_NUSOL_TENSOR

#include <torch/torch.h>
#include <nusol_tools.h>

namespace NuSol
{
    namespace Tensor
    {
        // masses = [W, Top, Neutrino]
        torch::Tensor BaseMatrix(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor masses); 

        std::tuple<torch::Tensor, torch::Tensor> Intersection(
                torch::Tensor A, torch::Tensor B, const double null); 

        std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor met_xy, 
                torch::Tensor masses, 
                torch::Tensor sigma); 
        
        std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor met_xy, 
                torch::Tensor masses, 
                torch::Tensor sigma, const double null); 

        std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
                torch::Tensor pmc_l1, torch::Tensor pmc_l2,
                torch::Tensor met_xy, 
                torch::Tensor masses, const double null); 
    }
}

#endif
