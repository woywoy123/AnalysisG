#include "cartesian.h"

torch::Tensor Transform::Tensors::Px(torch::Tensor pt, torch::Tensor phi)
{
        pt = pt.view({-1, 1});
        phi = phi.view({-1, 1}); 
        return pt * torch::cos(phi);
}

torch::Tensor Transform::Tensors::Py(torch::Tensor pt, torch::Tensor phi)
{
    pt = pt.view({-1, 1}); 
    phi = phi.view({-1, 1}); 
	return pt * torch::sin(phi);
}

torch::Tensor Transform::Tensors::Pz(torch::Tensor pt, torch::Tensor eta)
{
    pt = pt.view({-1, 1}); 
    eta = eta.view({-1, 1}); 
	return pt * torch::sinh(eta);
}

torch::Tensor Transform::Tensors::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi)
{
	torch::Tensor _px = Transform::Tensors::Px(pt, phi);
	torch::Tensor _py = Transform::Tensors::Py(pt, phi); 
	torch::Tensor _pz = Transform::Tensors::Pz(pt, eta); 
	return torch::cat({_px, _py, _pz}, -1);
}

torch::Tensor Transform::Tensors::PxPyPzE(torch::Tensor Pmu)
{
    Pmu = Pmu.view({-1, 4}); 
    torch::Tensor pt  = Pmu.index({torch::indexing::Slice(), 0}).view({-1, 1}); 
    torch::Tensor eta = Pmu.index({torch::indexing::Slice(), 1}).view({-1, 1}); 
    torch::Tensor phi = Pmu.index({torch::indexing::Slice(), 2}).view({-1, 1}); 
    torch::Tensor e   = Pmu.index({torch::indexing::Slice(), 3}).view({-1, 1}); 

    torch::Tensor _px = Transform::Tensors::Px(pt, phi);
	torch::Tensor _py = Transform::Tensors::Py(pt, phi); 
	torch::Tensor _pz = Transform::Tensors::Pz(pt, eta); 

	return torch::cat({_px, _py, _pz, e}, -1);
}
