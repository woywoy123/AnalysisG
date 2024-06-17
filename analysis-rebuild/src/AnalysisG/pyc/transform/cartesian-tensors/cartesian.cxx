#include "cartesian.h"

torch::Tensor Cclip(torch::Tensor inpt, int dim){ 
    return inpt.index({torch::indexing::Slice(), dim}); 
}

torch::Tensor transform::tensors::Px(torch::Tensor pt, torch::Tensor phi){
    pt = pt.view({-1, 1});
    phi = phi.view({-1, 1}); 
    return pt * torch::cos(phi);
}

torch::Tensor transform::tensors::Py(torch::Tensor pt, torch::Tensor phi){
    pt = pt.view({-1, 1}); 
    phi = phi.view({-1, 1}); 
    return pt * torch::sin(phi);
}

torch::Tensor transform::tensors::Pz(torch::Tensor pt, torch::Tensor eta){
    pt = pt.view({-1, 1}); 
    eta = eta.view({-1, 1}); 
    return pt * torch::sinh(eta);
}

torch::Tensor transform::tensors::PxPyPz(
        torch::Tensor pt, torch::Tensor eta, torch::Tensor phi
){
    torch::Tensor _px = transform::tensors::Px(pt, phi);
    torch::Tensor _py = transform::tensors::Py(pt, phi); 
    torch::Tensor _pz = transform::tensors::Pz(pt, eta); 
    return torch::cat({_px, _py, _pz}, -1);
}

torch::Tensor transform::tensors::PxPyPzE(
        torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
){
    return torch::cat({transform::tensors::PxPyPz(pt, eta, phi), e.view({-1, 1})}, -1);
}

torch::Tensor transform::tensors::Px(torch::Tensor pmu){
    torch::Tensor pt  = Cclip(pmu, 0);
    torch::Tensor phi = Cclip(pmu, 2);
    return transform::tensors::Px(pt, phi);  
}

torch::Tensor transform::tensors::Py(torch::Tensor pmu){
    torch::Tensor pt  = Cclip(pmu, 0); 
    torch::Tensor phi = Cclip(pmu, 2); 
    return transform::tensors::Py(pt, phi);
} 

torch::Tensor transform::tensors::Pz(torch::Tensor pmu){
    torch::Tensor pt  = Cclip(pmu, 0); 
    torch::Tensor eta = Cclip(pmu, 1); 
    return transform::tensors::Pz(pt, eta);
}

torch::Tensor transform::tensors::PxPyPz(torch::Tensor pmu){
    torch::Tensor pt  = Cclip(pmu, 0); 
    torch::Tensor eta = Cclip(pmu, 1); 
    torch::Tensor phi = Cclip(pmu, 2);  
    return transform::tensors::PxPyPz(pt, eta, phi);
}

torch::Tensor transform::tensors::PxPyPzE(torch::Tensor pmu){
    pmu = pmu.view({-1, 4}); 
    torch::Tensor pt  = Cclip(pmu, 0); 
    torch::Tensor eta = Cclip(pmu, 1); 
    torch::Tensor phi = Cclip(pmu, 2); 
    torch::Tensor e   = Cclip(pmu, 3).view({-1, 1}); 
    torch::Tensor _px = transform::tensors::Px(pt, phi);
    torch::Tensor _py = transform::tensors::Py(pt, phi); 
    torch::Tensor _pz = transform::tensors::Pz(pt, eta); 
    return torch::cat({_px, _py, _pz, e}, -1);
}
