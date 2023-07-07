#ifndef H_PYC_CUDA_
#define H_PYC_CUDA_

#include <transform/cartesian-cuda/cartesian.h>
#include <transform/polar-cuda/polar.h>
#include <physics/physics-cuda/physics.h>
#include <physics/physics-cuda/cartesian.h>
#include <physics/physics-cuda/polar.h>
#include <operators/operators-cuda/operators.h>
#include <nusol/nusol-cuda/nusol.h>

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
                torch::Tensor P (torch::Tensor px, torch::Tensor py, torch::Tensor pz);
                torch::Tensor Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
                torch::Tensor Beta (torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
                torch::Tensor M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
                torch::Tensor M (torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
                torch::Tensor Mt2(torch::Tensor pz, torch::Tensor e);
                torch::Tensor Mt (torch::Tensor pz, torch::Tensor e);
                torch::Tensor Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
                torch::Tensor DeltaR(torch::Tensor px1, torch::Tensor px2, torch::Tensor py1, torch::Tensor py2, torch::Tensor pz1, torch::Tensor pz2);

            }

            namespace combined 
            {
                torch::Tensor P2(torch::Tensor pmc); 
                torch::Tensor P(torch::Tensor pmc); 
                torch::Tensor Beta2(torch::Tensor pmc); 
                torch::Tensor Beta(torch::Tensor pmc); 
                torch::Tensor M2(torch::Tensor pmc);
                torch::Tensor M(torch::Tensor pmc);
                torch::Tensor Mt2(torch::Tensor pmc);
                torch::Tensor Mt(torch::Tensor pmc);
                torch::Tensor Theta(torch::Tensor pmc); 
                torch::Tensor DeltaR(torch::Tensor pmc1, torch::Tensor pmc2); 
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
                torch::Tensor M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
                torch::Tensor M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
                torch::Tensor Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e); 
                torch::Tensor Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e); 
                torch::Tensor Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi); 
                torch::Tensor DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2); 
            }

            namespace combined 
            {
                torch::Tensor P2(torch::Tensor pmu); 
                torch::Tensor P(torch::Tensor pmu); 
                torch::Tensor Beta2(torch::Tensor pmu); 
                torch::Tensor Beta(torch::Tensor pmu); 
                torch::Tensor M2(torch::Tensor pmu); 
                torch::Tensor M(torch::Tensor pmu); 
                torch::Tensor Mt2(torch::Tensor pmu); 
                torch::Tensor Mt(torch::Tensor pmu); 
                torch::Tensor Theta(torch::Tensor pmu); 
                torch::Tensor DeltaR(torch::Tensor pmu1, torch::Tensor pmu2); 
            }
        }
    }

    namespace operators
    {
        torch::Tensor Dot(torch::Tensor v1, torch::Tensor v2); 
        torch::Tensor Mul(torch::Tensor v1, torch::Tensor v2); 
        torch::Tensor CosTheta(torch::Tensor v1, torch::Tensor v2); 
        torch::Tensor SinTheta(torch::Tensor v1, torch::Tensor v2);
        torch::Tensor Rx(torch::Tensor angle); 
        torch::Tensor Ry(torch::Tensor angle); 
        torch::Tensor Rz(torch::Tensor angle); 
        torch::Tensor CoFactors(torch::Tensor matrix);
        torch::Tensor Determinant(torch::Tensor matrix); 
        torch::Tensor Inverse(torch::Tensor matrix); 
    }

    namespace nusol
    {
        torch::Tensor BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses); 
        torch::Tensor Nu(torch::Tensor pmc_b, torch::Tensor pmc_mu, 
                         torch::Tensor met_xy, torch::Tensor masses, torch::Tensor sigma); 
    }
}

#endif
