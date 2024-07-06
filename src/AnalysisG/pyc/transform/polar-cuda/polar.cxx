#include <polar-cuda.h>

torch::Tensor transform::cuda::Pclip(torch::Tensor inpt, int dim){ 
    return inpt.index({torch::indexing::Slice(), dim}); 
}

torch::Tensor transform::cuda::Pt(torch::Tensor px, torch::Tensor py){
    return _Pt(px, py);
} 

torch::Tensor transform::cuda::Pt(torch::Tensor pmc){
    torch::Tensor px = transform::cuda::Pclip(pmc, 0); 
    torch::Tensor py = transform::cuda::Pclip(pmc, 1); 
    return _Pt(px, py); 
}

torch::Tensor transform::cuda::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return _Eta(px, py, pz);
} 

torch::Tensor transform::cuda::Eta(torch::Tensor pmc){
    torch::Tensor px = transform::cuda::Pclip(pmc, 0); 
    torch::Tensor py = transform::cuda::Pclip(pmc, 1); 
    torch::Tensor pz = transform::cuda::Pclip(pmc, 2);
    return _Eta(px, py, pz); 
}

torch::Tensor transform::cuda::Phi(torch::Tensor px, torch::Tensor py){
    return _Phi(px, py);
} 

torch::Tensor transform::cuda::Phi(torch::Tensor pmc){
    torch::Tensor px = transform::cuda::Pclip(pmc, 0); 
    torch::Tensor py = transform::cuda::Pclip(pmc, 1); 
    return _Phi(px, py); 
}

torch::Tensor transform::cuda::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return _PtEtaPhi(px, py, pz);
}

torch::Tensor PtEtaPhi(torch::Tensor pmc){
    torch::Tensor px = transform::cuda::Pclip(pmc, 0); 
    torch::Tensor py = transform::cuda::Pclip(pmc, 1); 
    torch::Tensor pz = transform::cuda::Pclip(pmc, 2); 
    return _PtEtaPhi(px, py, pz); 
}

torch::Tensor transform::cuda::PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    px = px.view({-1, 1}); 
    py = py.view({-1, 1}); 
    pz = pz.view({-1, 1}); 
    e  =  e.view({-1, 1}); 
    return _PtEtaPhiE(torch::cat({px, py, pz, e}, -1));
}

torch::Tensor transform::cuda::PtEtaPhiE(torch::Tensor pmc){
    return _PtEtaPhiE(pmc);
}


