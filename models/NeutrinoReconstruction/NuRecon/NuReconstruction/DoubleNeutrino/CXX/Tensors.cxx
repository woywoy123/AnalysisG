#include "../../Physics/Tensors/Headers/PhysicsTensors.h"
#include "../../Physics/Floats/Headers/PhysicsFloats.h"
#include "../Headers/Floats.h"

std::vector<torch::Tensor> DoubleNu::Tensors::Init(
		torch::Tensor _b1, torch::Tensor _b2, 
		torch::Tensor _mu1, torch::Tensor _mu2, 
		torch::Tensor massTop, torch::Tensor massW, torch::Tensor massNu,
		torch::Tensor met, torch::Tensor phi)
{
	torch::Tensor _bC = PhysicsTensors::ToPxPyPzE(_b1); 
	torch::Tensor _muC = PhysicsTensors::ToPxPyPzE(_mu1); 





	//// ======== Algorithm ======= //
	//torch::Tensor Sol_ = NuSolutionTensors::AnalyticalSolutionsCartesian(_bC, _muC, massTop, massW, massNu); 
	//torch::Tensor S2_ = DoubleNu::Tensors::Sigma2(Sxx, Sxy, Syx, Syy); 
	//torch::Tensor V0_ = DoubleNu::Tensors::V0Polar(met, phi); 
	//torch::Tensor H_ = NuSolutionTensors::H_Algo(_b, _mu, Sol_);
	//
	//torch::Tensor dNu_ = V0_ - H_; 
	//torch::Tensor X_ = torch::matmul( torch::transpose(dNu_, 1, 2), S2_ ).matmul(dNu_); 
	//torch::Tensor D_ = NuSolutionTensors::Derivative(X_); 
	//torch::Tensor M_ = D_ + D_.transpose(1, 2);

	//// Protection against events where the dNU_ matrix is completely 0
	//torch::Tensor _SkipEvent = M_.sum({-1}).sum({-1}) == 0.; 
	//H_ = H_.index({_SkipEvent == false}); 
	//M_ = M_.index({_SkipEvent == false}); 
	//X_ = X_.index({_SkipEvent == false});

	//torch::Tensor C_ = NuSolutionTensors::UnitCircle(_mu).repeat({H_.sizes()[0], 1, 1}); 
	//Sol_ = NuSolutionTensors::Intersections(M_, C_, 1e-11);
	//
	//torch::Tensor chi2 = torch::sum(Sol_.view({-1, 6, 1, 3}) *X_.view({-1, 1, 3, 3}), 3);
	//chi2 = torch::sum(chi2.view({-1, 6, 1, 3}) * Sol_.view({-1, 1, 6, 3}), 3);
	//chi2 = chi2.diagonal(0, 1, 2);
	//std::tuple<torch::Tensor, torch::Tensor> idx = chi2.sort(1); 
	//torch::Tensor vals = std::get<0>(idx); 
	//torch::Tensor id = std::get<1>(idx);
	//Sol_ = Sol_.view({-1, 6, 3}); 
	//
	//torch::Tensor _t0 = torch::gather(Sol_.index({
	//			torch::indexing::Slice(), 
	//			torch::indexing::Slice(), 
	//			0}), 1, id);

	//torch::Tensor _t1 = torch::gather(Sol_.index({
	//			torch::indexing::Slice(), 
	//			torch::indexing::Slice(), 
	//			1}), 1, id);

	//torch::Tensor _t2 = torch::gather(Sol_.index({
	//			torch::indexing::Slice(), 
	//			torch::indexing::Slice(), 
	//			2}), 1, id); 


	//torch::Tensor msk = torch::argmax((vals != 0)*torch::arange(6, 0, -1, PhysicsTensors::Options(_mu)), -1, true); 
	//_t0 = torch::gather(_t0, 1, msk); 
	//_t1 = torch::gather(_t1, 1, msk); 
	//_t2 = torch::gather(_t2, 1, msk); 
	//vals = torch::gather(vals, 1, msk);
	//Sol_ = torch::cat({
	//		_t0.view({-1, 1}), 
	//		_t1.view({-1, 1}), 
	//		_t2.view({-1, 1})}, -1);

	//torch::Tensor NuSols = torch::sum(H_.view({-1, 3, 3})*Sol_.view({-1, 1, 3}), 2).view({-1, 3}); 
	return {}; 
}
