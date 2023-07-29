#ifndef H_NUSOL_TENSOR
#define H_NUSOL_TENSOR
#include <torch/torch.h>

namespace Tooling
{
    torch::TensorOptions MakeOp(torch::Tensor x); 
    std::map<std::string, torch::Tensor> GetMasses(torch::Tensor L, torch::Tensor masses); 
    torch::Tensor Pi_2(torch::Tensor x); 
    torch::Tensor x0(torch::Tensor pmc, torch::Tensor _pm2, torch::Tensor mH2, torch::Tensor mL2); 
    torch::Tensor Rotation(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor base); 
    torch::Tensor Sigma(torch::Tensor x, torch::Tensor sigma);
    torch::Tensor MET(torch::Tensor met_xy); 
    torch::Tensor Shape(torch::Tensor x, std::vector<int> diag); 
    torch::Tensor HorizontalVertical(torch::Tensor G); 
    torch::Tensor Parallel(torch::Tensor G, torch::Tensor CoF); 
    torch::Tensor Intersecting(torch::Tensor G, torch::Tensor g22, torch::Tensor CoF); 
}

namespace NuSol
{
    namespace Tensor
    {
        // masses = [W, Top, Neutrino]
        torch::Tensor BaseMatrix(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses); 

        std::tuple<torch::Tensor, torch::Tensor> Intersection(
                torch::Tensor A, torch::Tensor B, const double null); 

        std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor MET_xy, torch::Tensor masses, 
                torch::Tensor sigma); 
        
        std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor MET_xy, torch::Tensor masses, 
                torch::Tensor sigma, const double null); 

        std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
                torch::Tensor pmc_l1, torch::Tensor pmc_l2,
                torch::Tensor met_x , torch::Tensor met_y , 
                torch::Tensor masses, const double null); 
    }
}

#endif
