#ifndef H_PYC_CUDA_
#define H_PYC_CUDA_

#include <transform/cartesian-cuda/cartesian.h>
#include <transform/polar-cuda/polar.h>
#include <physics/physics-cuda/physics.h>
#include <physics/physics-cuda/cartesian.h>
#include <physics/physics-cuda/polar.h>

namespace pyc
{
    namespace transform
    {
        namespace separate
        {
            torch::Tensor Pt(torch::Tensor px, torch::Tensor py); 
            torch::Tensor Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
            torch::Tensor Phi(torch::Tensor px, torch::Tensor py); 
            torch::Tensor PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
            torch::Tensor PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);

            torch::Tensor Px(torch::Tensor pt, torch::Tensor phi); 
            torch::Tensor Py(torch::Tensor pt, torch::Tensor phi); 
            torch::Tensor Pz(torch::Tensor pt, torch::Tensor eta); 
            torch::Tensor PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
            torch::Tensor PxPyPzE(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
        }
        namespace combined
        {
            torch::Tensor Pt(torch::Tensor pmc); 
            torch::Tensor Eta(torch::Tensor pmc); 
            torch::Tensor Phi(torch::Tensor pmc); 
            torch::Tensor PtEtaPhi(torch::Tensor pmc); 
            torch::Tensor PtEtaPhiE(torch::Tensor pmc); 

            torch::Tensor Px(torch::Tensor pmu); 
            torch::Tensor Py(torch::Tensor pmu); 
            torch::Tensor Pz(torch::Tensor pmu); 
            torch::Tensor PxPyPz(torch::Tensor pmu); 
            torch::Tensor PxPyPzE(torch::Tensor pmu); 
        }
    }  

    namespace physics
    {
        namespace cartesian
        {
            namespace separate
            {
                torch::Tensor P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
                torch::Tensor P(torch::Tensor px, torch::Tensor py, torch::Tensor pz);
                torch::Tensor Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
                torch::Tensor Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
            }

            namespace combined 
            {
                torch::Tensor P2(torch::Tensor pmc); 
                torch::Tensor P(torch::Tensor pmc); 
                torch::Tensor Beta2(torch::Tensor pmc); 
                torch::Tensor Beta(torch::Tensor pmc); 
            }
        }
        namespace polar
        {

            namespace separate
            {
                torch::Tensor P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
                torch::Tensor P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
                torch::Tensor Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
                torch::Tensor Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
            }

            namespace combined 
            {
                torch::Tensor P2(torch::Tensor pmc); 
                torch::Tensor P(torch::Tensor pmc); 
                torch::Tensor Beta2(torch::Tensor pmc); 
                torch::Tensor Beta(torch::Tensor pmc); 
            }
        }
    }
}

#endif
