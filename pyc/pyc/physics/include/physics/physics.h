#ifndef CXX_PHYSICS_H
#define CXX_PHYSICS_H

#include <torch/torch.h>

namespace physics {
    namespace cartesian {
            torch::Tensor P2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz); 
            torch::Tensor P2(torch::Tensor* pmc); 
//            torch::Tensor P(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz); 
//            torch::Tensor P(torch::Tensor* pmc); 
//            torch::Tensor Beta2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e); 
//            torch::Tensor Beta2(torch::Tensor* pmc); 
//            torch::Tensor Beta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e); 
//            torch::Tensor Beta(torch::Tensor* pmc); 
//            torch::Tensor M2(torch::Tensor* pmc); 
//            torch::Tensor M2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e); 
//            torch::Tensor M(torch::Tensor* pmc); 
//            torch::Tensor M(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e); 
//            torch::Tensor Mt2(torch::Tensor* pmc); 
//            torch::Tensor Mt2(torch::Tensor* pz, torch::Tensor* e); 
//            torch::Tensor Mt(torch::Tensor* pmc); 
//            torch::Tensor Mt(torch::Tensor* pz, torch::Tensor* e); 
//            torch::Tensor Theta(torch::Tensor* pmc); 
//            torch::Tensor Theta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz); 
//            torch::Tensor DeltaR(torch::Tensor* pmc1, torch::Tensor* pmc2); 
//            torch::Tensor DeltaR(torch::Tensor* px1, torch::Tensor* px2, torch::Tensor* py1, torch::Tensor* py2, torch::Tensor* pz1, torch::Tensor* pz2); 
    }

//    namespace polar {
//            torch::Tensor P2(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi);
//            torch::Tensor P2(torch::Tensor* Pmu);
//            torch::Tensor P(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi);
//            torch::Tensor P(torch::Tensor* Pmu);
//            torch::Tensor Beta2(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* e);
//            torch::Tensor Beta2(torch::Tensor* pmu);
//            torch::Tensor Beta(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* e);
//            torch::Tensor Beta(torch::Tensor* pmu);
//            torch::Tensor M2(torch::Tensor* pmu);
//            torch::Tensor M2(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* e);
//            torch::Tensor M(torch::Tensor* pmu);
//            torch::Tensor M(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* e);
//            torch::Tensor Mt2(torch::Tensor* pmu);
//            torch::Tensor Mt2(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* e);
//            torch::Tensor Mt(torch::Tensor* pmu);
//            torch::Tensor Mt(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* e);
//            torch::Tensor Theta(torch::Tensor* pmu);
//            torch::Tensor Theta(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi);
//            torch::Tensor DeltaR(torch::Tensor* pmu1, torch::Tensor* pmu2);
//            torch::Tensor DeltaR(torch::Tensor* eta1, torch::Tensor* eta2, torch::Tensor* phi1, torch::Tensor* phi2);
//    }
}

#endif
