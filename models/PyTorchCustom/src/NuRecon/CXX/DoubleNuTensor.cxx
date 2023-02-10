#include "../Headers/NuSolTensor.h"

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

	torch::Tensor S_ = SingleNuTensor::V0(met_x, met_y) - NuSolTensors::UnitCircle(mNu2); 

	return S_;
}
