#include "../Headers/NuSolTensor.h"

torch::Tensor SingleNuTensor::Nu(torch::Tensor b, torch::Tensor mu, torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu)
{
	std::vector<torch::Tensor> b_ = NuSolTensors::_Format(b.view({-1, 4}), 4);
	torch::Tensor b_pt  = b_[0]; 
	torch::Tensor b_eta = b_[1]; 
	torch::Tensor b_phi = b_[2]; 
	torch::Tensor b_e   = b_[3]; 
	
	b_ = NuSolTensors::_Format(TransformTensors::PxPyPz(b_pt, b_eta, b_phi), 3); 
	torch::Tensor b_px = b_[0]; 
	torch::Tensor b_py = b_[1]; 
	torch::Tensor b_pz = b_[2]; 
	
	std::vector<torch::Tensor> mu_ = NuSolTensors::_Format(mu.view({-1, 4}), 4); 
	torch::Tensor mu_pt  = mu_[0]; 
	torch::Tensor mu_eta = mu_[1]; 
	torch::Tensor mu_phi = mu_[2]; 
	torch::Tensor mu_e   = mu_[3]; 

	mu_ = NuSolTensors::_Format(TransformTensors::PxPyPz(mu_pt, mu_eta, mu_phi), 3); 
	torch::Tensor mu_px = mu_[0]; 
	torch::Tensor mu_py = mu_[1]; 
	torch::Tensor mu_pz = mu_[2]; 
	
	torch::Tensor mT2 = mT.view({-1, 1}).pow(2); 
	torch::Tensor mW2 = mW.view({-1, 1}).pow(2); 
	torch::Tensor mNu2 = mNu.view({-1, 1}).pow(2);

	torch::Tensor sols_ = NuSolTensors::_Solutions(
			b_pt, b_eta, b_phi, b_e, b_px, b_py, b_pz, 
			mu_pt, mu_eta, mu_phi, mu_e, mu_px, mu_py, mu_pz,
			mT2, mW2, mNu2
	);

	torch::Tensor muP_ = PhysicsTensors::P(mu_px, mu_py, mu_pz); 
	torch::Tensor H_ = NuSolTensors::H_Matrix(sols_, b_px, b_py, b_pz, mu_phi, mu_pz, muP_); 	

	return H_; 


}

