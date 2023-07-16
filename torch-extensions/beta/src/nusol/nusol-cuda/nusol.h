#ifndef H_NUSOL_CUDA
#define H_NUSOL_CUDA

torch::Tensor _Shape_Matrix(torch::Tensor inpt, std::vector<long> diags); 
torch::Tensor _Expand_Matrix(torch::Tensor inpt, torch::Tensor source); 

torch::Tensor _Base_Matrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses_W_top_nu); 
torch::Tensor _Base_Matrix_H(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses_W_top_nu); 

torch::Tensor _DotMatrix(torch::Tensor MET_xy, torch::Tensor Sigma, torch::Tensor H);
torch::Tensor _Intersection(torch::Tensor A, torch::Tensor B); 

torch::Tensor _Nu(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor met_xy, torch::Tensor masses, torch::Tensor sigma); 

namespace NuSol
{
    namespace CUDA
    {
        // masses = [W, Top, Neutrino]
        const torch::Tensor BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses)
        {
            return _Base_Matrix(pmc_b, pmc_mu, masses);             
        }
      
        const torch::Tensor Intersection(torch::Tensor A, torch::Tensor B)
        {
            return _Intersection(A, B);  
        } 

        const torch::Tensor Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor MET_xy, 
                torch::Tensor masses, torch::Tensor sigma)
        {
            return _Nu(pmc_b, pmc_mu, MET_xy, masses, sigma); 
        }
    }
}

#endif
