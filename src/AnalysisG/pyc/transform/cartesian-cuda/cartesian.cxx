#include <cartesian-cuda.h>

torch::Tensor transform::cuda::Cclip(torch::Tensor inpt, int dim){
    return inpt.index({torch::indexing::Slice(), dim}); 
}

torch::Tensor transform::cuda::Px(torch::Tensor pt, torch::Tensor phi){
    return _Px(pt, phi);
} 

torch::Tensor transform::cuda::Px(torch::Tensor pmu){
    torch::Tensor pt = transform::cuda::Cclip(pmu, 0);
    torch::Tensor phi = transform::cuda::Cclip(pmu, 2);
    return _Px(pt, phi);  
}

torch::Tensor transform::cuda::Py(torch::Tensor pt, torch::Tensor phi){
    return _Py(pt, phi);
} 

torch::Tensor transform::cuda::Py(torch::Tensor pmu){
    torch::Tensor pt = transform::cuda::Cclip(pmu, 0); 
    torch::Tensor phi = transform::cuda::Cclip(pmu, 2); 
    return _Py(pt, phi);
} 

torch::Tensor transform::cuda::Pz(torch::Tensor pt, torch::Tensor eta){
    return _Pz(pt, eta);
}

torch::Tensor transform::cuda::Pz(torch::Tensor pmu){
    torch::Tensor pt = transform::cuda::Cclip(pmu, 0); 
    torch::Tensor eta = transform::cuda::Cclip(pmu, 1); 
    return _Pz(pt, eta);
}

torch::Tensor transform::cuda::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    return _PxPyPz(pt, eta, phi);
}

torch::Tensor transform::cuda::PxPyPz(torch::Tensor pmu){
    torch::Tensor pt = transform::cuda::Cclip(pmu, 0); 
    torch::Tensor eta = transform::cuda::Cclip(pmu, 1); 
    torch::Tensor phi = transform::cuda::Cclip(pmu, 2);  
    return _PxPyPz(pt, eta, phi);
}

torch::Tensor transform::cuda::PxPyPzE(
        torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
){
    pt = pt.view({-1, 1}); 
    eta = eta.view({-1, 1}); 
    phi = phi.view({-1, 1}); 
    e = e.view({-1, 1}); 
    return _PxPyPzE(torch::cat({pt, eta, phi, e}, -1));
} 

torch::Tensor transform::cuda::PxPyPzE(torch::Tensor Pmu){return _PxPyPzE(Pmu);}


