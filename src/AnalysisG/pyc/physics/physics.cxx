#include <physics/physics.h>
#include <cutils/utils.h>
#include <cmath>

torch::Tensor physics_::P2(torch::Tensor* pmc){
    torch::Tensor pmc_ = pmc -> index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 
    pmc_ = pmc_.pow(2);
    pmc_ = pmc_.sum({-1}); 
    return pmc_.view({-1, 1}); 
}

torch::Tensor physics_::P2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    std::vector<torch::Tensor> v = {*px, *py, *pz}; 
    torch::Tensor pmx = format(&v); 
    return physics_::P2(&pmx); 
}

torch::Tensor physics_::P(torch::Tensor* pmc){
    return torch::sqrt(physics_::P2(pmc));
}

torch::Tensor physics_::P(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pmx = format({px, py, pz}); 
    return physics_::P(&pmx); 
}

torch::Tensor physics_::Beta2(torch::Tensor* pmc){
    torch::Tensor e = pmc -> index({torch::indexing::Slice(), torch::indexing::Slice(3, 4)});
    return physics_::P2(pmc)/(e.pow(2)); 
}

torch::Tensor physics_::Beta2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({px, py, pz, e}); 
    return physics_::Beta2(&pmx);  
}

torch::Tensor physics_::Beta(torch::Tensor* pmc){
    return physics_::P(pmc)/pmc -> index({torch::indexing::Slice(), torch::indexing::Slice(3, 4)}); 
}

torch::Tensor physics_::Beta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({px, py, pz, e}); 
    return physics_::Beta(&pmx); 
}

torch::Tensor physics_::M2(torch::Tensor* pmc){
    torch::Tensor mass = pmc -> index({torch::indexing::Slice(), torch::indexing::Slice(3, 4)}); 
    mass = mass.pow(2); 
    mass -= physics_::P2(pmc); 
    return torch::relu(mass);
}

torch::Tensor physics_::M2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({px, py, pz, e}); 
    return physics_::M2(&pmx);
}    

torch::Tensor physics_::M(torch::Tensor* pmc){
    return torch::sqrt(physics_::M2(pmc));
}

torch::Tensor physics_::M(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({px, py, pz, e}); 
    return physics_::M(&pmx);     
}    

torch::Tensor physics_::Mt2(torch::Tensor* pmc){
    torch::Tensor pz = pmc -> index({torch::indexing::Slice(), torch::indexing::Slice(2, 3)}); 
    torch::Tensor e = pmc -> index({torch::indexing::Slice(), torch::indexing::Slice(3, 4)}); 
    return (torch::relu(e.pow(2) - pz.pow(2))).view({-1, 1}); 
}

torch::Tensor physics_::Mt2(torch::Tensor* pz, torch::Tensor* e){
    return  (torch::relu(e -> pow(2) - pz -> pow(2))).view({-1, 1});    
}    

torch::Tensor physics_::Mt(torch::Tensor* pmc){
    return torch::sqrt(physics_::Mt2(pmc)); 
}

torch::Tensor physics_::Mt(torch::Tensor* pz, torch::Tensor* e){
    return torch::sqrt(physics_::Mt2(pz, e));     
}

torch::Tensor physics_::Theta(torch::Tensor* pmc){
    torch::Tensor pz = pmc -> index({torch::indexing::Slice(), torch::indexing::Slice(2, 3)}); 
    return torch::acos( pz / physics_::P(pmc)); 
}

torch::Tensor physics_::Theta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pmx = format({px, py, pz}); 
    return torch::acos( pz -> view({-1, 1}) / physics_::P(&pmx) );  
}

torch::Tensor physics_::DeltaR(torch::Tensor* pmu1, torch::Tensor* pmu2){
    torch::Tensor d_eta = pmu1 -> index({torch::indexing::Slice(), torch::indexing::Slice(1, 3)}); 
    d_eta -= pmu2 -> index({torch::indexing::Slice(), torch::indexing::Slice(1, 3)});    
    
    torch::Tensor d_phi = d_eta.index({torch::indexing::Slice(), torch::indexing::Slice(1, 2)}); 
    d_eta = d_eta.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)}); 

    torch::Tensor pi = M_PI*torch::ones_like(d_eta); 
    d_phi = pi - torch::abs(torch::fmod(torch::abs(d_phi), 2*pi) - pi);   
    return torch::sqrt(d_eta.pow(2) + d_phi.pow(2)); 
}

torch::Tensor physics_::DeltaR(torch::Tensor* eta1, torch::Tensor* eta2, torch::Tensor* phi1, torch::Tensor* phi2){
    torch::Tensor d_eta = (*eta1) - (*eta2); 
    torch::Tensor d_phi = (*phi1) - (*phi2); 
    
    torch::Tensor pi = M_PI*torch::ones_like(d_eta); 
    d_phi = pi - torch::abs(torch::fmod(torch::abs(d_phi), 2*pi) - pi);   
    return torch::sqrt(d_eta.pow(2) + d_phi.pow(2)); 
}
