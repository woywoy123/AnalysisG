#include "polar.h"

torch::Tensor Transform::Tensors::Pt(torch::Tensor px, torch::Tensor py)
{
	return torch::sqrt( px.pow(2) + py.pow(2) ).view({-1, 1});
}

torch::Tensor Transform::Tensors::Phi(torch::Tensor px, torch::Tensor py)
{
	return torch::atan2( py, px ).view({-1, 1}); 
}

torch::Tensor Transform::Tensors::PtEta(torch::Tensor pt, torch::Tensor pz)
{
	return torch::asinh( pz / pt ).view({-1, 1}); 
}

torch::Tensor Transform::Tensors::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
{
	torch::Tensor pt = Transform::Tensors::Pt(px, py); 
	return Transform::Tensors::PtEta(pt, pz); 	
}

torch::Tensor Transform::Tensors::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
{
	torch::Tensor _pt = Transform::Tensors::Pt(px, py); 
	torch::Tensor _eta = Transform::Tensors::PtEta(_pt, pz.view({-1, 1})); 
	torch::Tensor _phi = Transform::Tensors::Phi(px, py);
	return torch::cat({_pt, _eta, _phi}, -1);
}

torch::Tensor Transform::Tensors::PtEtaPhiE(torch::Tensor Pmc)
{
    Pmc = Pmc.view({-1, 4}); 
    torch::Tensor px = Pmc.index({torch::indexing::Slice(), 0}).view({-1, 1}); 
    torch::Tensor py = Pmc.index({torch::indexing::Slice(), 1}).view({-1, 1}); 
    torch::Tensor pz = Pmc.index({torch::indexing::Slice(), 2}).view({-1, 1}); 
    torch::Tensor e  = Pmc.index({torch::indexing::Slice(), 3}).view({-1, 1}); 

    torch::Tensor ptetaphi = Transform::Tensors::PtEtaPhi(px, py, pz); 
    return torch::cat({ptetaphi, e}, -1); 
}
