#include "../Headers/NuSolTensor.h"

std::vector<torch::Tensor> NuSolTensors::_Format(torch::Tensor t, int dim)
{
	std::vector<torch::Tensor> _out; 
	for (int i = 0; i < dim; ++i)
	{
		_out.push_back(t.index({torch::indexing::Slice(), i}).view({-1, 1})); 
	}
	return _out; 
}


torch::Tensor NuSolTensors::Solutions(torch::Tensor b, torch::Tensor mu, torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu)
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

	return NuSolTensors::_Solutions(b_pt, b_eta, b_phi, b_e, b_px, b_py, b_pz, 
					mu_pt, mu_eta, mu_phi, mu_e, mu_px, mu_py, mu_pz,
					mT.view({-1, 1}).pow(2), 
					mW.view({-1, 1}).pow(2), 
					mNu.view({-1, 1}).pow(2)); 
}

torch::Tensor NuSolTensors::x0(torch::Tensor _pe, torch::Tensor _pm2, torch::Tensor mH2, torch::Tensor mL2)
{
	return -(mH2 - mL2 - _pm2)/(2*_pe); 
}

torch::Tensor NuSolTensors::_Solutions(
		torch::Tensor b_pt, torch::Tensor b_eta, torch::Tensor b_phi, torch::Tensor b_e,
		torch::Tensor b_px, torch::Tensor b_py, torch::Tensor b_pz,
		torch::Tensor mu_pt, torch::Tensor mu_eta, torch::Tensor mu_phi, torch::Tensor mu_e,
		torch::Tensor mu_px, torch::Tensor mu_py, torch::Tensor mu_pz,
		torch::Tensor massT2, torch::Tensor massW2, torch::Tensor massNu2)
{	
	torch::Tensor c_ = OperatorsTensors::CosTheta(torch::cat({b_px, b_py, b_pz}, -1), torch::cat({mu_px, mu_py, mu_pz}, -1)); 
	torch::Tensor s_ = OperatorsTensors::_SinTheta(c_);
	
	torch::Tensor x0p = NuSolTensors::x0(b_e, PhysicsTensors::M2(b_px, b_py, b_pz, b_e), massT2, massW2); 
	torch::Tensor x0 = NuSolTensors::x0(mu_e, PhysicsTensors::M2(mu_px, mu_py, mu_pz, mu_e), massW2, massNu2); 
	
	torch::Tensor bB = PhysicsTensors::Beta(b_px, b_py, b_pz, b_e); 
	torch::Tensor muB = PhysicsTensors::Beta(mu_px, mu_py, mu_pz, mu_e); 
	torch::Tensor muB2 = PhysicsTensors::Beta2(mu_px, mu_py, mu_pz, mu_e); 

	torch::Tensor Sx = (x0 * muB - PhysicsTensors::P(mu_px, mu_py, mu_pz) * ( 1 - muB2 )) / muB2; 
	torch::Tensor Sy = ( (x0p / bB) - c_ * Sx ) / s_; 
	
	torch::Tensor tmp_ = muB / bB; 
	torch::Tensor w = ( tmp_ - c_ ) / s_; 
	torch::Tensor w_ = ( - tmp_ - c_ ) / s_; 

	torch::Tensor O2 = w.pow(2) + 1 - muB2;
	torch::Tensor e2 = (massW2 - massNu2) * ( 1 - muB2 ); 
	
	tmp_ = Sx + w*Sy; 
	torch::Tensor x1 = Sx - tmp_ / O2; 
	torch::Tensor y1 = Sy - tmp_ * (w / O2); 
	torch::Tensor Z = torch::sqrt(torch::relu(x1.pow(2) * O2 - ( Sy - w*Sx ).pow(2) - ( massW2 - x0.pow(2) - e2 ))); 

	return torch::cat({ c_, s_, x0, x0p, Sx, Sy, w, w_, x1, y1, Z, O2, e2}, -1); 
}

torch::Tensor NuSolTensors::H_Matrix(torch::Tensor Sols_, torch::Tensor b_px, torch::Tensor b_py, torch::Tensor b_pz, torch::Tensor mu_phi, torch::Tensor mu_pz, torch::Tensor mu_P)
{
	torch::Tensor x1 = Sols_.index({torch::indexing::Slice(), 8}).view({-1, 1}); 
	torch::Tensor y1 = Sols_.index({torch::indexing::Slice(), 9}).view({-1, 1}); 
	torch::Tensor Z = Sols_.index({torch::indexing::Slice(), 10}).view({-1, 1}); 
	torch::Tensor Om = torch::sqrt(Sols_.index({torch::indexing::Slice(), 11}).view({-1, 1}));
	torch::Tensor w  = Sols_.index({torch::indexing::Slice(), 6}).view({-1, 1}); 

	torch::Tensor t0 = torch::zeros_like(x1); 
	torch::Tensor tmp_ = Z/Om; 
	torch::Tensor H_ = torch::cat({ tmp_, t0, x1 - mu_P, w*tmp_, t0, y1, t0, Z, t0 }, -1).view({-1, 3, 3}); 

	torch::Tensor theta_ = PhysicsTensors::Theta_(mu_P, mu_pz); 
	torch::Tensor Rz = OperatorsTensors::Rz(-mu_phi); 
	torch::Tensor Ry = OperatorsTensors::Ry(Constants::Pi_2(theta_) - theta_);
	
	torch::Tensor Rx = torch::matmul(Rz, torch::cat({b_px, b_py, b_pz}, -1).view({-1, 3, 1}));
	Rx = torch::matmul(Ry, Rx.view({-1, 3, 1})); 
	Rx = -torch::atan2(Rx.index({torch::indexing::Slice(), 2}), Rx.index({torch::indexing::Slice(), 1})).view({-1, 1}); 
	Rx = OperatorsTensors::Rx(Rx); 

	Rx = torch::transpose(Rx, 1, 2); 
	Ry = torch::transpose(Ry, 1, 2); 
	Rz = torch::transpose(Rz, 1, 2); 
	
	return torch::matmul(torch::matmul(Rz, torch::matmul(Ry, Rx)), H_); 
}
