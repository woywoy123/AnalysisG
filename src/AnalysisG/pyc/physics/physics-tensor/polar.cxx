#include <physics/polar.h>

torch::Tensor physics::tensors::polar::P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    torch::Tensor pmc = transform::tensors::PxPyPz(pt, eta, phi); 
    return physics::tensors::P2(pmc); 
} 

torch::Tensor physics::tensors::polar::P2(torch::Tensor pmu){
    torch::Tensor pmc = transform::tensors::PxPyPzE(pmu); 
    return physics::tensors::P2(pmc);
}

torch::Tensor physics::tensors::polar::P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
     torch::Tensor pmc = transform::tensors::PxPyPz(pt, eta, phi); 
     return physics::tensors::P(pmc); 
}

torch::Tensor physics::tensors::polar::P(torch::Tensor pmu){
    torch::Tensor pmc = transform::tensors::PxPyPzE(pmu); 
    return physics::tensors::P(pmc);
}

torch::Tensor physics::tensors::polar::Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ 
    torch::Tensor pmc = transform::tensors::PxPyPzE(pt, eta, phi, e); 
    return physics::tensors::Beta2(pmc); 
}

torch::Tensor physics::tensors::polar::Beta2(torch::Tensor pmu){
    torch::Tensor pmc = transform::tensors::PxPyPzE(pmu); 
    return physics::tensors::Beta2(pmc); 
}

torch::Tensor physics::tensors::polar::Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ 
    torch::Tensor pmc = transform::tensors::PxPyPzE(pt, eta, phi, e); 
    return physics::tensors::Beta(pmc); 
}

torch::Tensor physics::tensors::polar::Beta(torch::Tensor pmu){
    torch::Tensor pmc = transform::tensors::PxPyPzE(pmu); 
    return physics::tensors::Beta(pmc); 
}

torch::Tensor physics::tensors::polar::M2(torch::Tensor pmu){ 
    torch::Tensor pmc = transform::tensors::PxPyPzE(pmu); 
    return physics::tensors::M2(pmc); 
}

torch::Tensor physics::tensors::polar::M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ 
    torch::Tensor pmc = transform::tensors::PxPyPzE(pt, eta, phi, e); 
    return physics::tensors::M2(pmc); 
}

torch::Tensor physics::tensors::polar::M(torch::Tensor pmu){ 
    torch::Tensor pmc = transform::tensors::PxPyPzE(pmu); 
    return physics::tensors::M(pmc); 
}

torch::Tensor physics::tensors::polar::M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ 
    torch::Tensor pmc = transform::tensors::PxPyPzE(pt, eta, phi, e); 
    return physics::tensors::M(pmc); 
}

torch::Tensor physics::tensors::polar::Mt2(torch::Tensor pmu){ 
    torch::Tensor pmc = transform::tensors::PxPyPzE(pmu); 
    return physics::tensors::Mt2(pmc);
}

torch::Tensor physics::tensors::polar::Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){ 
    torch::Tensor pz = transform::tensors::Pz(pt, eta); 
    return physics::tensors::Mt2(pz, e); 
}

torch::Tensor physics::tensors::polar::Mt(torch::Tensor pmu){ 
    torch::Tensor pmc = transform::tensors::PxPyPzE(pmu); 
    return physics::tensors::Mt(pmc); 
}

torch::Tensor physics::tensors::polar::Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){ 
    torch::Tensor pz = transform::tensors::Pz(pt, eta); 
    return physics::tensors::Mt(pz, e); 
}

torch::Tensor physics::tensors::polar::Theta(torch::Tensor pmu){
    torch::Tensor pmc = transform::tensors::PxPyPzE(pmu); 
    return physics::tensors::Theta(pmc); 
}

torch::Tensor physics::tensors::polar::Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    torch::Tensor pmc = transform::tensors::PxPyPz(torch::cat({pt.view({-1, 1}), eta.view({-1, 1}), phi.view({-1, 1})}, -1)); 
    return physics::tensors::Theta(pmc); 
}

torch::Tensor physics::tensors::polar::DeltaR(torch::Tensor pmu1, torch::Tensor pmu2){
    return physics::tensors::DeltaR(pmu1, pmu2); 
}

torch::Tensor physics::tensors::polar::DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2){
    return physics::tensors::DeltaR(eta1, eta2, phi1, phi2); 
}
