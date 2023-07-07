#ifndef H_NUSOL_CUDA
#define H_NUSOL_CUDA

torch::Tensor _Base_Matrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses_W_top_nu); 
torch::Tensor _Nu_Matrix(torch::Tensor MET_xy, torch::Tensor Sigma, torch::Tensor H);
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
            torch::Tensor H = _Base_Matrix(pmc_b, pmc_mu, masses); 
            return _Nu_Matrix(MET_xy, sigma, H); 
        }
    }
}

#endif
