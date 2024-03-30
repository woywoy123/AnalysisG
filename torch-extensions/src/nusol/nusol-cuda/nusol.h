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

namespace NuSol
{
    namespace CUDA
    {
        // masses = [W, Top, Neutrino]
        inline torch::Tensor BaseMatrix(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor masses)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(pmc_b.get_device()); 
            torch::Tensor out = _Base_Matrix(pmc_b, pmc_mu, masses);             
            c10::cuda::set_device(current_device);
            return out; 

        }
      
        inline std::tuple<torch::Tensor, torch::Tensor> Intersection(
                torch::Tensor A, torch::Tensor B, const double null)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(A.get_device()); 
            std::tuple<torch::Tensor, torch::Tensor> out = _Intersection(A, B, null);  
            c10::cuda::set_device(current_device);
            return out;
        } 

        inline std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor met_xy, torch::Tensor masses, 
                torch::Tensor sigma)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(pmc_b.get_device()); 
            std::map<std::string, torch::Tensor> output = _Nu(pmc_b, pmc_mu, met_xy, masses, sigma); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                torch::Tensor met_xy, 
                torch::Tensor masses, 
                torch::Tensor sigma, const double null)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(pmc_b.get_device()); 
            std::map<std::string, torch::Tensor> output = _Nu(
                    pmc_b, pmc_mu, met_xy, 
                    masses, sigma, null
            ); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
                torch::Tensor pmc_l1, torch::Tensor pmc_l2,
                torch::Tensor met_xy, 
                torch::Tensor masses, const double null)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(pmc_b1.get_device()); 
            std::map<std::string, torch::Tensor> output = _NuNu(
                    pmc_b1, pmc_b2, 
                    pmc_l1, pmc_l2, 
                    met_xy, masses, null
            ); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline std::map<std::string, torch::Tensor> combinatorial(
                torch::Tensor edge_index, torch::Tensor batch, 
                torch::Tensor pmc, torch::Tensor pid, torch::Tensor met_xy, 
                const double mass_top, const double mass_W, const double mass_nu,
                const double top_up_down, const double w_up_down, const double null)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(pmc.get_device()); 
            const double mass_top_l = mass_top*top_up_down; 
            const double mass_top_u = mass_top*(1 + (1-top_up_down)); 

            const double mass_w_l = mass_W*w_up_down; 
            const double mass_w_u = mass_W*(1 + (1-w_up_down)); 

            std::map<std::string, torch::Tensor> output = _CombinatorialCartesian(
                edge_index, batch, pmc, pid, met_xy, 
                mass_top_l, mass_top_u, mass_w_l, mass_w_u, mass_nu, null
            );

            c10::cuda::set_device(current_device);
            return output;
        }

        namespace Polar
        {
            inline std::map<std::string, torch::Tensor> Nu(
                    torch::Tensor pmu_b, torch::Tensor pmu_mu, 
                    torch::Tensor met_phi, torch::Tensor masses, 
                    torch::Tensor sigma, const double null)
            {
                const auto current_device = c10::cuda::current_device();
                c10::cuda::set_device(pmu_b.get_device()); 
                std::map<std::string, torch::Tensor> output = _NuPolar(
                        pmu_b, pmu_mu, 
                        met_phi, masses, 
                        sigma, null
                ); 
                c10::cuda::set_device(current_device);
                return output;
            }

            inline std::map<std::string, torch::Tensor> Nu(
                torch::Tensor pt_b, torch::Tensor eta_b, torch::Tensor phi_b, torch::Tensor e_b, 
                torch::Tensor pt_mu, torch::Tensor eta_mu, torch::Tensor phi_mu, torch::Tensor e_mu, 
                torch::Tensor met, torch::Tensor phi, torch::Tensor masses, 
                torch::Tensor sigma, const double null)
            {
                const auto current_device = c10::cuda::current_device();
                c10::cuda::set_device(met.get_device()); 

                std::map<std::string, torch::Tensor> output = _NuPolar(
                        pt_b , eta_b , phi_b , e_b, 
                        pt_mu, eta_mu, phi_mu, e_mu, 
                        met, phi, masses, sigma, null); 
                c10::cuda::set_device(current_device);
                return output;
            }


            inline std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor pmu_b1 , torch::Tensor pmu_b2, 
                torch::Tensor pmu_mu1, torch::Tensor pmu_mu2, 
                torch::Tensor met_phi, torch::Tensor masses, 
                const double null)
            {
                const auto current_device = c10::cuda::current_device();
                c10::cuda::set_device(pmu_b1.get_device()); 
                std::map<std::string, torch::Tensor> output = _NuNuPolar(pmu_b1, pmu_b2, pmu_mu1, pmu_mu2, met_phi, masses, null);
                c10::cuda::set_device(current_device);
                return output;
            }

            inline std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor pt_b1, torch::Tensor eta_b1, torch::Tensor phi_b1, torch::Tensor e_b1, 
                torch::Tensor pt_b2, torch::Tensor eta_b2, torch::Tensor phi_b2, torch::Tensor e_b2, 

                torch::Tensor pt_mu1, torch::Tensor eta_mu1, torch::Tensor phi_mu1, torch::Tensor e_mu1, 
                torch::Tensor pt_mu2, torch::Tensor eta_mu2, torch::Tensor phi_mu2, torch::Tensor e_mu2, 

                torch::Tensor met, torch::Tensor phi, 
                torch::Tensor masses, const double null)
            {
                const auto current_device = c10::cuda::current_device();
                c10::cuda::set_device(pt_b1.get_device()); 
                std::map<std::string, torch::Tensor> output = _NuNuPolar(
                        pt_b1,  eta_b1,  phi_b1,  e_b1 , pt_b2,  eta_b2,  phi_b2,  e_b2, 
                        pt_mu1, eta_mu1, phi_mu1, e_mu1, pt_mu2, eta_mu2, phi_mu2, e_mu2, 
                        met, phi, masses, null);
                c10::cuda::set_device(current_device);
                return output;
            }
        }

        namespace Cartesian
        {
            inline std::map<std::string, torch::Tensor> Nu(
                    torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                    torch::Tensor met_xy, torch::Tensor masses, 
                    torch::Tensor sigma, const double null)
            {
                const auto current_device = c10::cuda::current_device();
                c10::cuda::set_device(pmc_b.get_device()); 
                std::map<std::string, torch::Tensor> output = _NuCart(pmc_b, pmc_mu, met_xy, masses, sigma, null); 
                c10::cuda::set_device(current_device);
                return output;
            }

            inline std::map<std::string, torch::Tensor> Nu(
                    torch::Tensor px_b, torch::Tensor py_b, torch::Tensor pz_b, torch::Tensor e_b, 
                    torch::Tensor px_mu, torch::Tensor py_mu, torch::Tensor pz_mu, torch::Tensor e_mu, 
                    torch::Tensor metx, torch::Tensor mety, torch::Tensor masses, 
                    torch::Tensor sigma, const double null)
            {
                const auto current_device = c10::cuda::current_device();
                c10::cuda::set_device(px_b.get_device()); 
                std::map<std::string, torch::Tensor> output = _NuCart(
                        px_b, py_b, pz_b, e_b, px_mu, py_mu, pz_mu, e_mu, 
                        metx, mety, masses, sigma, null); 
                c10::cuda::set_device(current_device);
                return output;
            }

            inline std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
                torch::Tensor pmc_mu1, torch::Tensor pmc_mu2,
                torch::Tensor met_xy, torch::Tensor masses, const double null)
            {
                const auto current_device = c10::cuda::current_device();
                c10::cuda::set_device(pmc_b1.get_device()); 
                std::map<std::string, torch::Tensor> output = _NuNuCart(
                        pmc_b1, pmc_b2, 
                        pmc_mu1, pmc_mu2, 
                        met_xy, masses, null
                );
                c10::cuda::set_device(current_device);
                return output;
            }

            inline std::map<std::string, torch::Tensor> NuNu(
                torch::Tensor px_b1, torch::Tensor py_b1, torch::Tensor pz_b1, torch::Tensor e_b1, 
                torch::Tensor px_b2, torch::Tensor py_b2, torch::Tensor pz_b2, torch::Tensor e_b2, 

                torch::Tensor px_mu1, torch::Tensor py_mu1, torch::Tensor pz_mu1, torch::Tensor e_mu1, 
                torch::Tensor px_mu2, torch::Tensor py_mu2, torch::Tensor pz_mu2, torch::Tensor e_mu2, 

                torch::Tensor metx, torch::Tensor mety, torch::Tensor masses, const double null)
            {
                const auto current_device = c10::cuda::current_device();
                c10::cuda::set_device(px_b1.get_device()); 
                std::map<std::string, torch::Tensor> output = _NuNuCart(
                        px_b1,  py_b1,  pz_b1,  e_b1, px_b2,  py_b2,  pz_b2,  e_b2, 
                        px_mu1, py_mu1, pz_mu1, e_mu1, px_mu2, py_mu2, pz_mu2, e_mu2, 
                        metx, mety, masses, null
                ); 
                c10::cuda::set_device(current_device);
                return output;
            }
        }
    }
}

#endif
