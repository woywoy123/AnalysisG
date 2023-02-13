#include "../Headers/NuSolTensor.h"

torch::Tensor DoubleNuTensor::N(torch::Tensor H)
{
	torch::Tensor H_ = torch::clone(H); 
	H_.index_put_({torch::indexing::Slice(), 2, torch::indexing::Slice()}, 0); 
	H_.index_put_({torch::indexing::Slice(), 2, 2}, 1);
	H_ = torch::inverse(H_); 
	torch::Tensor H_T = torch::transpose(H_, 1, 2); 
	H_T = torch::matmul(H_T, NuSolTensors::UnitCircle(H_)); 
	return torch::matmul(H_T, H_); 
}

torch::Tensor DoubleNuTensor::NuNu(
			torch::Tensor b, torch::Tensor b_, 
			torch::Tensor mu, torch::Tensor mu_, 
			torch::Tensor met, torch::Tensor phi, 
			torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu)
{

	// ---- Polar Version of Particles ---- //
	std::vector<torch::Tensor> b_P = NuSolTensors::_Format(b.view({-1, 4}), 4);
	std::vector<torch::Tensor> b__P = NuSolTensors::_Format(b_.view({-1, 4}), 4);
	std::vector<torch::Tensor> mu_P = NuSolTensors::_Format(mu.view({-1, 4}), 4); 
	std::vector<torch::Tensor> mu__P = NuSolTensors::_Format(mu_.view({-1, 4}), 4); 
	
	// ---- Cartesian Version of Particles ---- //
	std::vector<torch::Tensor> b_C = NuSolTensors::_Format(TransformTensors::PxPyPz(b_P[0], b_P[1], b_P[2]), 3); 
	std::vector<torch::Tensor> b__C = NuSolTensors::_Format(TransformTensors::PxPyPz(b__P[0], b__P[1], b__P[2]), 3); 
	std::vector<torch::Tensor> mu_C = NuSolTensors::_Format(TransformTensors::PxPyPz(mu_P[0], mu_P[1], mu_P[2]), 3); 
	std::vector<torch::Tensor> mu__C = NuSolTensors::_Format(TransformTensors::PxPyPz(mu__P[0], mu__P[1], mu__P[2]), 3); 
	
	// ----- Some useful values ------ //
	torch::Tensor muP_ = PhysicsTensors::P(mu_C[0], mu_C[1], mu_C[2]); 
	torch::Tensor mu_e   = mu_P[3]; 
	torch::Tensor b_e = b_P[3]; 

	torch::Tensor muP__ = PhysicsTensors::P(mu__C[0], mu__C[1], mu__C[2]); 
	torch::Tensor mu__e   = mu__P[3]; 
	torch::Tensor b__e = b__P[3]; 

	// ---- Cartesian Version of Event Met ---- //
	torch::Tensor met_x = TransformTensors::Px(met, phi); 
	torch::Tensor met_y = TransformTensors::Py(met, phi);
	
	// ---- Precalculate the Mass Squared ---- //
	torch::Tensor mT2 = mT.view({-1, 1}).pow(2); 
	torch::Tensor mW2 = mW.view({-1, 1}).pow(2); 
	torch::Tensor mNu2 = mNu.view({-1, 1}).pow(2);
	
	// Starting the algorithm 
	torch::Tensor sols_ = NuSolTensors::_Solutions(b_C, mu_C, b_e, mu_e, mT2, mW2, mNu2);
	torch::Tensor sols__ = NuSolTensors::_Solutions(b__C, mu__C, b__e, mu__e, mT2, mW2, mNu2);
	
	torch::Tensor H_ = NuSolTensors::H_Matrix(sols_, b_C, mu_P[2], mu_C[2], muP_); 
	torch::Tensor H__ = NuSolTensors::H_Matrix(sols__, b__C, mu__P[2], mu__C[2], muP__); 
	torch::Tensor SkipEvent = (torch::det(H_) != 0)*(torch::det(H__) != 0);

	H_ = H_.index({SkipEvent}); 
	H__ = H__.index({SkipEvent}); 
	met_x = met_x.index({SkipEvent}); 
	met_y = met_y.index({SkipEvent});
	
	torch::Tensor N_ = DoubleNuTensor::N(H_.index({SkipEvent})); 
	torch::Tensor N__ = DoubleNuTensor::N(H__.index({SkipEvent})); 

	torch::Tensor S_ = NuSolTensors::V0(met_x.index({SkipEvent}), met_y.index({SkipEvent})) - NuSolTensors::UnitCircle(mNu2.index({SkipEvent})); 
	torch::Tensor n_ = torch::matmul(torch::matmul(S_.transpose(1, 2), N__), S_); 
	
	return NuSolTensors::Intersections(N_, n_);
}
