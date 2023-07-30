#ifndef H_NUSOL_CUDA
#define H_NUSOL_CUDA

torch::Tensor _Base_Matrix(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, 
        torch::Tensor masses_W_top_nu); 

std::tuple<torch::Tensor, torch::Tensor> _Intersection(
        torch::Tensor A, torch::Tensor B, const double null); 

std::map<std::string, torch::Tensor> _Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, 
        torch::Tensor met_xy, torch::Tensor masses, 
        torch::Tensor sigma); 

std::map<std::string, torch::Tensor> _Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, 
        torch::Tensor met_xy, torch::Tensor masses, 
        torch::Tensor sigma, const double null); 

std::map<std::string, torch::Tensor> _NuNu(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
        torch::Tensor pmc_l1, torch::Tensor pmc_l2,
        torch::Tensor met_xy, torch::Tensor masses, 
        const double null);


namespace NuSol
{
    namespace CUDA
    {
        // masses = [W, Top, Neutrino]
        const torch::Tensor BaseMatrix(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor masses)
        {
            return _Base_Matrix(pmc_b, pmc_mu, masses);             
        }
      
        const std::tuple<torch::Tensor, torch::Tensor> Intersection(
                torch::Tensor A, torch::Tensor B, const double null)
        {
            return _Intersection(A, B, null);  
        } 

        const std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor met_xy, torch::Tensor masses, 
                torch::Tensor sigma)
        {
            return _Nu(pmc_b, pmc_mu, met_xy, masses, sigma); 
        }

        const std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor met_xy, 
                torch::Tensor masses, 
                torch::Tensor sigma, const double null)
        {
            return _Nu(pmc_b, pmc_mu, met_xy, masses, sigma, null); 
        }

        const std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
                torch::Tensor pmc_l1, torch::Tensor pmc_l2,
                torch::Tensor met_xy, 
                torch::Tensor masses, const double null)
        {
            return _NuNu(pmc_b1, pmc_b2, pmc_l1, pmc_l2, met_xy, masses, null); 
        }
    }
}

#endif
