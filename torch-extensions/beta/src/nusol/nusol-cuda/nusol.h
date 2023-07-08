#include <operators.h>

#ifndef H_NUSOL_CUDA
#define H_NUSOL_CUDA

torch::Tensor _Shape_Matrix(torch::Tensor inpt, std::vector<long> diags); 
torch::Tensor _Expand_Matrix(torch::Tensor inpt, torch::Tensor source); 

torch::Tensor _Base_Matrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses_W_top_nu); 
torch::Tensor _Base_Matrix_H(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses_W_top_nu); 

torch::Tensor _DotMatrix(torch::Tensor MET_xy, torch::Tensor Sigma, torch::Tensor H);
torch::Tensor _NuNu_Matrix(torch::Tensor MET_xy); 

namespace NuSol
{
    namespace CUDA
    {
        // masses = [W, Top, Neutrino]
        const torch::Tensor BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses)
        {
            return _Base_Matrix(pmc_b, pmc_mu, masses);             
        }
       
        const torch::Tensor Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor MET_xy, 
                torch::Tensor masses, torch::Tensor sigma)
        {
            torch::Tensor H = _Base_Matrix_H(pmc_b, pmc_mu, masses); 
            torch::Tensor shape = _Shape_Matrix(H, {0, 0, 1});
            sigma = _Expand_Matrix(H, sigma.view({-1, 2, 2})) + shape; 
            sigma = Operators::CUDA::Inverse(sigma) - shape;
            return _DotMatrix(MET_xy, H, sigma); 
        }
    }
}

#endif
