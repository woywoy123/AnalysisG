#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <stdio.h>
#include <cuda.h>

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

std::map<std::string, torch::Tensor> _NuPolar(
        torch::Tensor pmu_b, torch::Tensor pmu_mu, 
        torch::Tensor met_phi, torch::Tensor masses, 
        torch::Tensor sigma, const double null); 

std::map<std::string, torch::Tensor> _NuPolar(
        torch::Tensor pt_b, torch::Tensor eta_b, torch::Tensor phi_b, torch::Tensor e_b, 
        torch::Tensor pt_mu, torch::Tensor eta_mu, torch::Tensor phi_mu, torch::Tensor e_mu, 
        torch::Tensor met, torch::Tensor phi, torch::Tensor masses, 
        torch::Tensor sigma, const double null); 

std::map<std::string, torch::Tensor> _NuCart(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, 
        torch::Tensor met_xy, torch::Tensor masses, 
        torch::Tensor sigma, const double null); 

std::map<std::string, torch::Tensor> _NuCart(
        torch::Tensor px_b, torch::Tensor py_b, torch::Tensor pz_b, torch::Tensor e_b, 
        torch::Tensor px_mu, torch::Tensor py_mu, torch::Tensor pz_mu, torch::Tensor e_mu, 
        torch::Tensor metx, torch::Tensor mety, torch::Tensor masses, 
        torch::Tensor sigma, const double null); 


std::map<std::string, torch::Tensor> _NuNu(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
        torch::Tensor pmc_l1, torch::Tensor pmc_l2,
        torch::Tensor met_xy, torch::Tensor masses, 
        const double null);

std::map<std::string, torch::Tensor> _NuNuPolar(
        torch::Tensor pmu_b1 , torch::Tensor pmu_b2, 
        torch::Tensor pmu_mu1, torch::Tensor pmu_mu2, 
        torch::Tensor met_phi, torch::Tensor masses, 
        const double null); 

std::map<std::string, torch::Tensor> _NuNuPolar(
        torch::Tensor pt_b1, torch::Tensor eta_b1, torch::Tensor phi_b1, torch::Tensor e_b1, 
        torch::Tensor pt_b2, torch::Tensor eta_b2, torch::Tensor phi_b2, torch::Tensor e_b2, 

        torch::Tensor pt_mu1, torch::Tensor eta_mu1, torch::Tensor phi_mu1, torch::Tensor e_mu1, 
        torch::Tensor pt_mu2, torch::Tensor eta_mu2, torch::Tensor phi_mu2, torch::Tensor e_mu2, 

        torch::Tensor met, torch::Tensor phi, 
        torch::Tensor masses, const double null); 

std::map<std::string, torch::Tensor> _NuNuCart(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
        torch::Tensor pmc_mu1, torch::Tensor pmc_mu2,
        torch::Tensor met_xy, torch::Tensor masses, const double null); 

std::map<std::string, torch::Tensor> _NuNuCart(
        torch::Tensor px_b1, torch::Tensor py_b1, torch::Tensor pz_b1, torch::Tensor e_b1, 
        torch::Tensor px_b2, torch::Tensor py_b2, torch::Tensor pz_b2, torch::Tensor e_b2, 

        torch::Tensor px_mu1, torch::Tensor py_mu1, torch::Tensor pz_mu1, torch::Tensor e_mu1, 
        torch::Tensor px_mu2, torch::Tensor py_mu2, torch::Tensor pz_mu2, torch::Tensor e_mu2, 

        torch::Tensor metx, torch::Tensor mety, torch::Tensor masses, const double null); 

std::map<std::string, torch::Tensor> _CombinatorialCartesian(
        torch::Tensor edge_index, torch::Tensor batch, 
        torch::Tensor pmc, torch::Tensor pid, torch::Tensor met_xy, 
        const double mass_top_l, const double mass_top_u, const double mass_w_l, const double mass_w_u, 
        const double mass_nu, const double null
); 

namespace nusol {
    namespace cuda {
        // masses = [W, Top, Neutrino]
        torch::Tensor BaseMatrix(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor masses); 
        
        std::tuple<torch::Tensor, torch::Tensor> Intersection(
                torch::Tensor A, torch::Tensor B, const double null); 
        
        std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor met_xy, torch::Tensor masses, 
                torch::Tensor sigma); 
        
        std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor met_xy, torch::Tensor masses, 
                torch::Tensor sigma, const double null); 
        
        std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
                torch::Tensor pmc_l1, torch::Tensor pmc_l2,
                torch::Tensor met_xy, 
                torch::Tensor masses, const double null); 
        
        std::map<std::string, torch::Tensor> combinatorial(
                torch::Tensor edge_index, torch::Tensor batch, 
                torch::Tensor pmc, torch::Tensor pid, torch::Tensor met_xy, 
                const double mass_top, const double mass_W, const double mass_nu,
                const double top_up_down, const double w_up_down, const double null); 
    
        namespace polar {
            std::map<std::string, torch::Tensor> Nu(
                    torch::Tensor pmu_b, torch::Tensor pmu_mu, 
                    torch::Tensor met_phi, torch::Tensor masses, 
                    torch::Tensor sigma, const double null); 

            std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pt_b, torch::Tensor eta_b, torch::Tensor phi_b, torch::Tensor e_b, 
                torch::Tensor pt_mu, torch::Tensor eta_mu, torch::Tensor phi_mu, torch::Tensor e_mu, 
                torch::Tensor met, torch::Tensor phi, torch::Tensor masses, 
                torch::Tensor sigma, const double null); 

            std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor pmu_b1 , torch::Tensor pmu_b2, 
                torch::Tensor pmu_mu1, torch::Tensor pmu_mu2, 
                torch::Tensor met_phi, torch::Tensor masses, 
                const double null); 

            std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor pt_b1, torch::Tensor eta_b1, torch::Tensor phi_b1, torch::Tensor e_b1, 
                torch::Tensor pt_b2, torch::Tensor eta_b2, torch::Tensor phi_b2, torch::Tensor e_b2, 

                torch::Tensor pt_mu1, torch::Tensor eta_mu1, torch::Tensor phi_mu1, torch::Tensor e_mu1, 
                torch::Tensor pt_mu2, torch::Tensor eta_mu2, torch::Tensor phi_mu2, torch::Tensor e_mu2, 

                torch::Tensor met, torch::Tensor phi, 
                torch::Tensor masses, const double null); 
        }

        namespace cartesian {
            std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor met_xy, torch::Tensor masses, 
                torch::Tensor sigma, const double null); 
        
            std::map<std::string, torch::Tensor> Nu(
                torch::Tensor px_b, torch::Tensor py_b, torch::Tensor pz_b, torch::Tensor e_b, 
                torch::Tensor px_mu, torch::Tensor py_mu, torch::Tensor pz_mu, torch::Tensor e_mu, 
                torch::Tensor metx, torch::Tensor mety, torch::Tensor masses, 
                torch::Tensor sigma, const double null); 
        
            std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
                torch::Tensor pmc_mu1, torch::Tensor pmc_mu2,
                torch::Tensor met_xy, torch::Tensor masses, const double null); 
        
            std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor px_b1, torch::Tensor py_b1, torch::Tensor pz_b1, torch::Tensor e_b1, 
                torch::Tensor px_b2, torch::Tensor py_b2, torch::Tensor pz_b2, torch::Tensor e_b2, 
        
                torch::Tensor px_mu1, torch::Tensor py_mu1, torch::Tensor pz_mu1, torch::Tensor e_mu1, 
                torch::Tensor px_mu2, torch::Tensor py_mu2, torch::Tensor pz_mu2, torch::Tensor e_mu2, 
        
                torch::Tensor metx, torch::Tensor mety, torch::Tensor masses, const double null); 
        }
    }
}

#endif
