#include <physics/physics-cuda.h>
#include <physics/cartesian-cuda.h>
#include <physics/polar-cuda.h>

torch::Tensor physics::cuda::P2(torch::Tensor pmc){return _P2(pmc);}
torch::Tensor physics::cuda::P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz){return _P2(px, py, pz);}
torch::Tensor physics::cuda::P(torch::Tensor pmc){return _P(pmc);}
torch::Tensor physics::cuda::P(torch::Tensor px, torch::Tensor py, torch::Tensor pz){return _P(px, py, pz);}
torch::Tensor physics::cuda::Beta2(torch::Tensor pmc){return _Beta2(pmc);}
torch::Tensor physics::cuda::Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){return _Beta2(px, py, pz, e);}
torch::Tensor physics::cuda::Beta(torch::Tensor pmc){return _Beta(pmc);}
torch::Tensor physics::cuda::Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){return _Beta(px, py, pz, e);}
torch::Tensor physics::cuda::M2(torch::Tensor pmc){return _M2(pmc);}
torch::Tensor physics::cuda::M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){return _M2(px, py, pz, e);}
torch::Tensor physics::cuda::M(torch::Tensor pmc){return _M(pmc);}
torch::Tensor physics::cuda::M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){return _M(px, py, pz, e);}
torch::Tensor physics::cuda::Mt2(torch::Tensor pmc){return _Mt2(pmc);}
torch::Tensor physics::cuda::Mt2(torch::Tensor pz, torch::Tensor e){return _Mt2(pz, e);}
torch::Tensor physics::cuda::Mt(torch::Tensor pmc){return _Mt(pmc);}
torch::Tensor physics::cuda::Mt(torch::Tensor pz, torch::Tensor e){return _Mt(pz, e);}
torch::Tensor physics::cuda::Theta(torch::Tensor pmc){return _Theta(pmc);}
torch::Tensor physics::cuda::Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){return _Theta(px, py, pz);}
torch::Tensor physics::cuda::DeltaR(torch::Tensor pmu1, torch::Tensor pmu2){return _DeltaR(pmu1, pmu2);}
torch::Tensor physics::cuda::DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2){return _DeltaR(eta1, eta2, phi1, phi2);}

torch::Tensor physics::cuda::cartesian::P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ 
    return physics::cuda::P2(px, py, pz); 
}

torch::Tensor physics::cuda::cartesian::P2(torch::Tensor pmc){
    return physics::cuda::P2(pmc); 
}

torch::Tensor physics::cuda::cartesian::P(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ 
    return physics::cuda::P(px, py, pz); 
}

torch::Tensor physics::cuda::cartesian::P(torch::Tensor pmc){
    return physics::cuda::P(pmc); 
}

torch::Tensor physics::cuda::cartesian::Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ 
    return physics::cuda::Beta2(px, py, pz, e); 
}

torch::Tensor physics::cuda::cartesian::Beta2(torch::Tensor pmc){
    return physics::cuda::Beta2(pmc); 
}

torch::Tensor physics::cuda::cartesian::Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ 
    return physics::cuda::Beta(px, py, pz, e); 
}

torch::Tensor physics::cuda::cartesian::Beta(torch::Tensor pmc){
    return physics::cuda::Beta(pmc); 
}

torch::Tensor physics::cuda::cartesian::M2(torch::Tensor pmc){ 
    return physics::cuda::M2(pmc); 
}

torch::Tensor physics::cuda::cartesian::M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ 
    return physics::cuda::M2(px, py, pz, e); 
}

torch::Tensor physics::cuda::cartesian::M(torch::Tensor pmc){ 
    return physics::cuda::M(pmc); 
}

torch::Tensor physics::cuda::cartesian::M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ 
    return physics::cuda::M(px, py, pz, e); 
}

torch::Tensor physics::cuda::cartesian::Mt2(torch::Tensor pmc){ 
    return physics::cuda::Mt2(pmc);
}

torch::Tensor physics::cuda::cartesian::Mt2(torch::Tensor pz, torch::Tensor e){ 
    return physics::cuda::Mt2(pz, e); 
}

torch::Tensor physics::cuda::cartesian::Mt(torch::Tensor pmc){ 
    return physics::cuda::Mt(pmc); 
}

torch::Tensor physics::cuda::cartesian::Mt(torch::Tensor pz, torch::Tensor e){ 
    return physics::cuda::Mt(pz, e); 
}

torch::Tensor physics::cuda::cartesian::Theta(torch::Tensor pmc){
    return physics::cuda::Theta(pmc); 
}

torch::Tensor physics::cuda::cartesian::Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return physics::cuda::Theta(px, py, pz); 
}

torch::Tensor physics::cuda::cartesian::DeltaR(torch::Tensor pmc1, torch::Tensor pmc2){
    const unsigned int len = pmc1.size(0);
    torch::Tensor pmu1 = transform::cuda::PtEtaPhiE(torch::cat({pmc1, pmc2}, 0)); 
    torch::Tensor pmu2 = pmu1.index({torch::indexing::Slice({len, len*2+1})}).view({-1, 4});
    pmu1 = pmu1.index({torch::indexing::Slice({0, len})}).view({-1, 4});
    return physics::cuda::DeltaR(pmu1, pmu2); 
}

torch::Tensor physics::cuda::cartesian::DeltaR(
        torch::Tensor px1, torch::Tensor px2, 
        torch::Tensor py1, torch::Tensor py2, 
        torch::Tensor pz1, torch::Tensor pz2
){
    const unsigned int len = px1.size(0);
    torch::Tensor pmu1 = transform::cuda::PtEtaPhi(
            torch::cat({px1, px2}, 0), 
            torch::cat({py1, py2}, 0), 
            torch::cat({pz1, pz2}, 0)
    ); 
    torch::Tensor pmu2 = pmu1.index({torch::indexing::Slice({len, len*2+1})}).view({-1, 3});
    pmu1 = pmu1.index({torch::indexing::Slice({0, len})}).view({-1, 3});
    return physics::cuda::DeltaR(pmu1, pmu2); 
}

torch::Tensor physics::cuda::polar::P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    return physics::cuda::P2(transform::cuda::PxPyPz(pt, eta, phi)); 
} 

torch::Tensor physics::cuda::polar::P2(torch::Tensor Pmu){
    return physics::cuda::P2(transform::cuda::PxPyPzE(Pmu));
}

torch::Tensor physics::cuda::polar::P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
     return physics::cuda::P(transform::cuda::PxPyPz(pt, eta, phi)); 
}

torch::Tensor physics::cuda::polar::P(torch::Tensor Pmu){
    return physics::cuda::P(transform::cuda::PxPyPzE(Pmu));
}

torch::Tensor physics::cuda::polar::Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ 
    return physics::cuda::Beta2(transform::cuda::PxPyPzE(pt, eta, phi, e)); 
}

torch::Tensor physics::cuda::polar::Beta2(torch::Tensor pmu){
    return physics::cuda::Beta2(transform::cuda::PxPyPzE(pmu)); 
}

torch::Tensor physics::cuda::polar::Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ 
    return physics::cuda::Beta(transform::cuda::PxPyPzE(pt, eta, phi, e)); 
}

torch::Tensor physics::cuda::polar::Beta(torch::Tensor pmu){
    return physics::cuda::Beta(transform::cuda::PxPyPzE(pmu)); 
}

torch::Tensor physics::cuda::polar::M2(torch::Tensor pmu){ 
    return physics::cuda::M2(transform::cuda::PxPyPzE(pmu)); 
}

torch::Tensor physics::cuda::polar::M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ 
    return physics::cuda::M2(transform::cuda::PxPyPzE(pt, eta, phi, e)); 
}

torch::Tensor physics::cuda::polar::M(torch::Tensor pmu){ 
    return physics::cuda::M(transform::cuda::PxPyPzE(pmu)); 
}

torch::Tensor physics::cuda::polar::M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ 
    return physics::cuda::M(transform::cuda::PxPyPzE(pt, eta, phi, e)); 
}

torch::Tensor physics::cuda::polar::Mt2(torch::Tensor pmu){ 
    return physics::cuda::Mt2(transform::cuda::PxPyPzE(pmu));
}

torch::Tensor physics::cuda::polar::Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){ 
    return physics::cuda::Mt2(transform::cuda::Pz(pt, eta), e); 
}

torch::Tensor physics::cuda::polar::Mt(torch::Tensor pmu){ 
    return physics::cuda::Mt(transform::cuda::PxPyPzE(pmu)); 
}

torch::Tensor physics::cuda::polar::Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){ 
    return physics::cuda::Mt(transform::cuda::Pz(pt, eta), e); 
}

torch::Tensor physics::cuda::polar::Theta(torch::Tensor pmu){
    return physics::cuda::Theta(transform::cuda::PxPyPzE(pmu)); 
}

torch::Tensor physics::cuda::polar::Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    return physics::cuda::Theta(transform::cuda::PxPyPz(torch::cat({pt.view({-1, 1}), eta.view({-1, 1}), phi.view({-1, 1})}, -1))); 
}

torch::Tensor physics::cuda::polar::DeltaR(torch::Tensor pmu1, torch::Tensor pmu2){
    return physics::cuda::DeltaR(pmu1, pmu2); 
}

torch::Tensor physics::cuda::polar::DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2){
    return physics::cuda::DeltaR(eta1, eta2, phi1, phi2); 
}
