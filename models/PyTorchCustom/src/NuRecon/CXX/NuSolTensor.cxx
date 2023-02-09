#include "../Headers/NuSolTensor.h"
#include <cstddef>

std::vector<torch::Tensor> NuSolTensors::_Format(torch::Tensor t, int dim)
{
	std::vector<torch::Tensor> _out; 
	for (int i = 0; i < dim; ++i)
	{
		_out.push_back(t.index({torch::indexing::Slice(), i}).view({-1, 1})); 
	}
	return _out; 
}

torch::Tensor NuSolTensors::x0(torch::Tensor _pe, torch::Tensor _pm2, torch::Tensor mH2, torch::Tensor mL2)
{
	return -(mH2 - mL2 - _pm2)/(2*_pe); 
}

torch::Tensor NuSolTensors::_UnitCircle(torch::Tensor x, std::vector<int> diag)
{
	torch::TensorOptions op = torch::TensorOptions().device(x.device()).dtype(x.dtype()); 
	torch::Tensor circ = torch::diag(torch::tensor(diag, op)).view({-1, 3, 3});
	std::vector<torch::Tensor> _out(x.size(0), circ); 
	return torch::cat(_out, 0); 
}

torch::Tensor NuSolTensors::Derivative(torch::Tensor x)
{
	torch::Tensor _o = OperatorsTensors::Rz(Constants::Pi_2(x));
	return _o.matmul(_UnitCircle(x, {1, 1, 0}));
}


torch::Tensor NuSolTensors::_Solutions(
		std::vector<torch::Tensor> b_C, std::vector<torch::Tensor> mu_C, 
		torch::Tensor b_e, torch::Tensor mu_e,
		torch::Tensor massT2, torch::Tensor massW2, torch::Tensor massNu2)
{	
	
	torch::Tensor bB = PhysicsTensors::Beta(b_C[0], b_C[1], b_C[2], b_e); 
	torch::Tensor muB = PhysicsTensors::Beta(mu_C[0], mu_C[1], mu_C[2], mu_e); 
	torch::Tensor muB2 = PhysicsTensors::Beta2(mu_C[0], mu_C[1], mu_C[2], mu_e); 

	torch::Tensor x0p = NuSolTensors::x0(b_e, PhysicsTensors::M2(b_C[0], b_C[1], b_C[2], b_e), massT2, massW2); 
	torch::Tensor x0 = NuSolTensors::x0(mu_e, PhysicsTensors::M2(mu_C[0], mu_C[1], mu_C[2], mu_e), massW2, massNu2); 
	
	torch::Tensor c_ = OperatorsTensors::CosTheta(torch::cat(b_C, -1), torch::cat(mu_C, -1)); 
	torch::Tensor s_ = OperatorsTensors::_SinTheta(c_);

	torch::Tensor tmp_ = muB / bB; 
	torch::Tensor w_ = ( - tmp_ - c_ ) / s_; 
	torch::Tensor w = ( tmp_ - c_ ) / s_; 
	
	torch::Tensor O2 = w.pow(2) + 1 - muB2;
	torch::Tensor e2 = (massW2 - massNu2) * ( 1 - muB2 ); 
	
	torch::Tensor Sx = (x0 * muB - PhysicsTensors::P(mu_C[0], mu_C[1], mu_C[2]) * ( 1 - muB2 )) / muB2; 
	torch::Tensor Sy = ( (x0p / bB) - c_ * Sx ) / s_; 
	
	tmp_ = Sx + w*Sy; 
	torch::Tensor x1 = Sx - tmp_ / O2; 
	torch::Tensor y1 = Sy - tmp_ * (w / O2); 

	torch::Tensor Z = torch::sqrt(torch::relu(x1.pow(2) * O2 - ( Sy - w*Sx ).pow(2) - ( massW2 - x0.pow(2) - e2 ))); 

	return torch::cat({ c_, s_, x0, x0p, Sx, Sy, w, w_, x1, y1, Z, O2, e2}, -1); 
}

torch::Tensor NuSolTensors::H_Matrix(torch::Tensor Sols_, std::vector<torch::Tensor> b_C, torch::Tensor mu_phi, torch::Tensor mu_pz, torch::Tensor mu_P)
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
	
	torch::Tensor Rx = torch::matmul(Rz, torch::cat(b_C, -1).view({-1, 3, 1}));
	Rx = torch::matmul(Ry, Rx.view({-1, 3, 1})); 
	Rx = -torch::atan2(Rx.index({torch::indexing::Slice(), 2}), Rx.index({torch::indexing::Slice(), 1})).view({-1, 1}); 
	Rx = OperatorsTensors::Rx(Rx); 

	Rx = torch::transpose(Rx, 1, 2); 
	Ry = torch::transpose(Ry, 1, 2); 
	Rz = torch::transpose(Rz, 1, 2); 
	
	return torch::matmul(torch::matmul(Rz, torch::matmul(Ry, Rx)), H_); 
}

















torch::Tensor NuSolTensors::Solutions(torch::Tensor b, torch::Tensor mu, torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu)
{
	std::vector<torch::Tensor> b_P = _Format(b.view({-1, 4}), 4);
	std::vector<torch::Tensor> b_C = _Format(TransformTensors::PxPyPz(b_P[0], b_P[1], b_P[2]), 3); 
	torch::Tensor b_e = b_P[3]; 
	
	std::vector<torch::Tensor> mu_P = _Format(mu.view({-1, 4}), 4); 
	std::vector<torch::Tensor> mu_C = _Format(TransformTensors::PxPyPz(mu_P[0], mu_P[1], mu_P[2]), 3); 
	torch::Tensor mu_e   = mu_P[3]; 
	
	mT = mT.view({-1, 1}).pow(2);
	mW = mW.view({-1, 1}).pow(2);
	mNu = mNu.view({-1, 1}).pow(2); 

	return NuSolTensors::_Solutions(b_C, mu_C, b_P[3], mu_P[3], mT, mW, mNu); 
}

torch::Tensor NuSolTensors::UnitCircle(torch::Tensor x){return _UnitCircle(x, {1, 1, -1});}
