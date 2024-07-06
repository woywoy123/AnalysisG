#include <polar.h>

torch::Tensor Pclip(torch::Tensor inpt, int dim){ 
    return inpt.index({torch::indexing::Slice(), dim}); 
}

torch::Tensor transform::tensors::Pt(torch::Tensor px, torch::Tensor py){
	return torch::sqrt( px.pow(2) + py.pow(2) ).view({-1, 1});
}

torch::Tensor transform::tensors::Phi(torch::Tensor px, torch::Tensor py){
	return torch::atan2( py, px ).view({-1, 1}); 
}

torch::Tensor transform::tensors::PtEta(torch::Tensor pt, torch::Tensor pz){
	return torch::asinh( pz / pt ).view({-1, 1}); 
}

torch::Tensor transform::tensors::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
	torch::Tensor pt = transform::tensors::Pt(px, py); 
	return transform::tensors::PtEta(pt, pz); 	
}

torch::Tensor transform::tensors::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
	torch::Tensor _pt  = transform::tensors::Pt(px, py); 
	torch::Tensor _eta = transform::tensors::PtEta(_pt, pz.view({-1, 1})); 
	torch::Tensor _phi = transform::tensors::Phi(px, py);
	return torch::cat({_pt, _eta, _phi}, -1);
}

torch::Tensor transform::tensors::PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return torch::cat({transform::tensors::PtEtaPhi(px, py, pz), e.view({-1, 1})}, -1);
}

torch::Tensor transform::tensors::Pt(torch::Tensor pmc){
	return transform::tensors::Pt(Pclip(pmc, 0), Pclip(pmc, 1));
}

torch::Tensor transform::tensors::Eta(torch::Tensor pmc){
	return transform::tensors::Eta(Pclip(pmc, 0), Pclip(pmc, 1), Pclip(pmc, 2));
}

torch::Tensor transform::tensors::Phi(torch::Tensor pmc){
    return transform::tensors::Phi(Pclip(pmc, 0), Pclip(pmc, 1));
} 

torch::Tensor transform::tensors::PtEtaPhi(torch::Tensor pmc){
    return transform::tensors::PtEtaPhi(Pclip(pmc, 0), Pclip(pmc, 1), Pclip(pmc, 2)); 
}

torch::Tensor transform::tensors::PtEtaPhiE(torch::Tensor pmc){
    return torch::cat({transform::tensors::PtEtaPhi(pmc), Pclip(pmc, 3).view({-1, 1})}, -1);
}
