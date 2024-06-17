#include <physics/cartesian.h>

torch::Tensor physics::tensors::cartesian::P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ 
    return physics::tensors::P2(px, py, pz); 
}

torch::Tensor physics::tensors::cartesian::P2(torch::Tensor pmc){
    return physics::tensors::P2(pmc); 
}

torch::Tensor physics::tensors::cartesian::P(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ 
    return physics::tensors::P(px, py, pz); 
}

torch::Tensor physics::tensors::cartesian::P(torch::Tensor pmc){
    return physics::tensors::P(pmc); 
}

torch::Tensor physics::tensors::cartesian::Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ 
    return physics::tensors::Beta2(px, py, pz, e); 
}

torch::Tensor physics::tensors::cartesian::Beta2(torch::Tensor pmc){
    return physics::tensors::Beta2(pmc); 
}

torch::Tensor physics::tensors::cartesian::Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ 
    return physics::tensors::Beta(px, py, pz, e); 
}

torch::Tensor physics::tensors::cartesian::Beta(torch::Tensor pmc){
    return physics::tensors::Beta(pmc); 
}

torch::Tensor physics::tensors::cartesian::M2(torch::Tensor pmc){ 
    return physics::tensors::M2(pmc); 
}

torch::Tensor physics::tensors::cartesian::M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ 
    return physics::tensors::M2(px, py, pz, e); 
}

torch::Tensor physics::tensors::cartesian::M(torch::Tensor pmc){ 
    return physics::tensors::M(pmc); 
}

torch::Tensor physics::tensors::cartesian::M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ 
    return physics::tensors::M(px, py, pz, e); 
}

torch::Tensor physics::tensors::cartesian::Mt2(torch::Tensor pmc){ 
    return physics::tensors::Mt2(pmc);
}

torch::Tensor physics::tensors::cartesian::Mt2(torch::Tensor pz, torch::Tensor e){ 
    return physics::tensors::Mt2(pz, e); 
}

torch::Tensor physics::tensors::cartesian::Mt(torch::Tensor pmc){ 
    return physics::tensors::Mt(pmc); 
}

torch::Tensor physics::tensors::cartesian::Mt(torch::Tensor pz, torch::Tensor e){ 
    return physics::tensors::Mt(pz, e); 
}

torch::Tensor physics::tensors::cartesian::Theta(torch::Tensor pmc){
    return physics::tensors::Theta(pmc); 
}

torch::Tensor physics::tensors::cartesian::Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return physics::tensors::Theta(px, py, pz); 
}

torch::Tensor physics::tensors::cartesian::DeltaR(torch::Tensor pmc1, torch::Tensor pmc2){
    const unsigned int len = pmc1.size(0);
    torch::Tensor pmu1 = transform::tensors::PtEtaPhiE(torch::cat({pmc1, pmc2}, 0)); 
    torch::Tensor pmu2 = pmu1.index({torch::indexing::Slice({len, len*2+1})}).view({-1, 4});
    pmu1 = pmu1.index({torch::indexing::Slice({0, len})}).view({-1, 4});
    return physics::tensors::DeltaR(pmu1, pmu2); 
}

torch::Tensor physics::tensors::cartesian::DeltaR(
        torch::Tensor px1, torch::Tensor px2, 
        torch::Tensor py1, torch::Tensor py2, 
        torch::Tensor pz1, torch::Tensor pz2
){
    const unsigned int len = px1.size(0);
    torch::Tensor pmu1 = transform::tensors::PtEtaPhi(
            torch::cat({px1, px2}, 0), 
            torch::cat({py1, py2}, 0), 
            torch::cat({pz1, pz2}, 0)); 
    torch::Tensor pmu2 = pmu1.index({torch::indexing::Slice({len, len*2+1})}).view({-1, 3});
    pmu1 = pmu1.index({torch::indexing::Slice({0, len})}).view({-1, 3});
    return physics::tensors::DeltaR(pmu1, pmu2); 
}


