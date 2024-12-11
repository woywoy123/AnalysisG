#include <transform/transform.h>
#include <cutils/utils.h>

torch::Tensor transform_::Px(torch::Tensor* pt, torch::Tensor* phi){
        return pt -> view({-1, 1}) * torch::cos(phi -> view({-1, 1}));
}

torch::Tensor transform_::Py(torch::Tensor* pt, torch::Tensor* phi){
    return pt -> view({-1, 1}) * torch::sin(phi -> view({-1, 1})); 
}

torch::Tensor transform_::Pz(torch::Tensor* pt, torch::Tensor* eta){
    return pt -> view({-1, 1}) * torch::sinh(eta -> view({-1, 1})); 
}

torch::Tensor transform_::PxPyPz(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi){
	torch::Tensor _px = transform_::Px(pt, phi);
	torch::Tensor _py = transform_::Py(pt, phi); 
	torch::Tensor _pz = transform_::Pz(pt, eta); 
	return torch::cat({_px, _py, _pz}, -1);
}

torch::Tensor transform_::PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* e){
    return torch::cat({transform_::PxPyPz(pt, eta, phi), e -> view({-1, 1})}, -1);
}

torch::Tensor transform_::PxPyPz(torch::Tensor* pmu){
    torch::Tensor pt  = clip(pmu, 0); 
    torch::Tensor eta = clip(pmu, 1); 
    torch::Tensor phi = clip(pmu, 2);
    if (pmu -> size({-1}) < 4){return transform_::PxPyPz(&pt, &eta, &phi);}
    return torch::cat({transform_::PxPyPz(&pt, &eta, &phi), clip(pmu, 3).view({-1, 1})}, {-1});
}

torch::Tensor transform_::PxPyPzE(torch::Tensor* pmu){
    torch::Tensor pt  = clip(pmu, 0).view({-1, 1}); 
    torch::Tensor eta = clip(pmu, 1).view({-1, 1}); 
    torch::Tensor phi = clip(pmu, 2).view({-1, 1}); 
    torch::Tensor e   = clip(pmu, 3).view({-1, 1}); 

    torch::Tensor _px = transform_::Px(&pt, &phi);
    torch::Tensor _py = transform_::Py(&pt, &phi); 
    torch::Tensor _pz = transform_::Pz(&pt, &eta); 

    return torch::cat({_px, _py, _pz, e}, -1);
}


torch::Tensor transform_::Pt(torch::Tensor* px, torch::Tensor* py){
    return torch::sqrt( px -> pow(2) + py -> pow(2) ).view({-1, 1});
}

torch::Tensor transform_::Phi(torch::Tensor* px, torch::Tensor* py){
    return torch::atan2( *py, *px ).view({-1, 1}); 
}

torch::Tensor transform_::Phi(torch::Tensor* pmc){
    torch::Tensor px = clip(pmc, 0); 
    torch::Tensor py = clip(pmc, 1); 
    return transform_::Phi(&px, &py);  
}

torch::Tensor transform_::PtEta(torch::Tensor* pt, torch::Tensor* pz){
    return torch::asinh( pz -> view({-1, 1}) / pt -> view({-1, 1}) ); 
}

torch::Tensor transform_::Eta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pt = transform_::Pt(px, py); 
    return transform_::PtEta(&pt, pz); 	
}

torch::Tensor transform_::Eta(torch::Tensor* pmc){
    torch::Tensor px = clip(pmc, 0); 
    torch::Tensor py = clip(pmc, 1);
    torch::Tensor pz = clip(pmc, 2);     
    return transform_::Eta(&px, &py, &pz);
}

torch::Tensor transform_::PtEtaPhi(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor _pt  = transform_::Pt(px, py); 
    torch::Tensor _eta = transform_::PtEta(&_pt, pz); 
    torch::Tensor _phi = transform_::Phi(px, py);

    return torch::cat({_pt, _eta, _phi}, -1);
}

torch::Tensor transform_::PtEtaPhi(torch::Tensor* pmc){
    torch::Tensor px = clip(pmc, 0); 
    torch::Tensor py = clip(pmc, 1);
    torch::Tensor pz = clip(pmc, 2);     
    if (pmc -> size({-1}) < 4){return transform_::PtEtaPhi(&px, &py, &pz);}
    return torch::cat({transform_::PtEtaPhi(&px, &py, &pz), clip(pmc, 3).view({-1, 1})}, {-1});
}

torch::Tensor transform_::PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    return torch::cat({transform_::PtEtaPhi(px, py, pz), e -> view({-1, 1})}, -1);
}

torch::Tensor transform_::PtEtaPhiE(torch::Tensor* pmc){
    return transform_::PtEtaPhi(pmc); 
}

