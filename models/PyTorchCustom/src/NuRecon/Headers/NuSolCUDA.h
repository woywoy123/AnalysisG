#ifndef H_NUSOL_CUDA
#define H_NUSOL_CUDA 

#include <iostream>
#include <torch/extension.h>
#include <iostream>
#include "../../Physics/Headers/CUDA.h"
#include "../../Transform/Headers/ToCartesianCUDA.h"
#include "../../Operators/Headers/CUDA.h"

torch::Tensor _Solutions(
		torch::Tensor _muP2, torch::Tensor _bP2, 
		torch::Tensor _mu_e, torch::Tensor _b_e, 
		torch::Tensor _cos, torch::Tensor _sin, 
		torch::Tensor mT2, torch::Tensor mW2, torch::Tensor mNu2);

torch::Tensor _H_Matrix(
		torch::Tensor x1, torch::Tensor y1, torch::Tensor Z, 
		torch::Tensor Om, torch::Tensor w, 
		std::vector<torch::Tensor> b_C, 
		torch::Tensor mu_phi, torch::Tensor mu_pz, torch::Tensor mu_P, 
		torch::Tensor mu_theta, torch::Tensor Rx, torch::Tensor Ry, torch::Tensor Rz); 


torch::Tensor _H_Matrix(torch::Tensor sols, torch::Tensor mu_P); 
torch::Tensor _Pi_2(torch::Tensor V);
torch::Tensor _Unit(torch::Tensor v, std::vector<int> diag); 
torch::Tensor _Factorization(torch::Tensor G); 
torch::Tensor _Factorization(torch::Tensor G, torch::Tensor Q, torch::Tensor Cofactors); 
torch::Tensor _SwapXY(torch::Tensor G, torch::Tensor Q);
std::vector<torch::Tensor> _EllipseLines(torch::Tensor Lines, torch::Tensor Q, torch::Tensor A); 

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


namespace NuSolCUDA
{
	const void _CheckTensors(std::vector<torch::Tensor> T){for (torch::Tensor x : T){CHECK_INPUT(x);}}
	const std::vector<torch::Tensor> _Format(torch::Tensor t, int dim)
	{
		std::vector<torch::Tensor> _out; 
		for (int i = 0; i < dim; ++i)
		{
			_out.push_back((t.index({torch::indexing::Slice(), i}).view({-1, 1}).contiguous())); 
		}
		return _out; 
	}

	const torch::Tensor Solutions(
		std::vector<torch::Tensor> b_P, std::vector<torch::Tensor> b_C, 
		std::vector<torch::Tensor> mu_P, std::vector<torch::Tensor> mu_C, 
		torch::Tensor massT2, torch::Tensor massW2, torch::Tensor massNu2)
	{
		_CheckTensors({massT2, massW2, massNu2});	
		torch::Tensor _muP2 = PhysicsCUDA::P2(mu_C[0], mu_C[1], mu_C[2]);	
		torch::Tensor _bP2 = PhysicsCUDA::P2(b_C[0], b_C[1], b_C[2]); 

		torch::Tensor _cos = OperatorsCUDA::CosTheta(
				torch::cat({b_C[0], b_C[1], b_C[2]}, -1), 
				torch::cat({mu_C[0], mu_C[1], mu_C[2]}, -1)
		);

		torch::Tensor _sin = OperatorsCUDA::_SinTheta(_cos); 

		return _Solutions(_muP2, _bP2, mu_P[3], b_P[3], _cos, _sin, massT2, massW2, massNu2);
	}

	const torch::Tensor V0(torch::Tensor metx, torch::Tensor mety)
	{
		torch::Tensor matrix = torch::cat({metx, mety}, -1).view({-1, 2});
		matrix = torch::pad(matrix, {0, 1, 0, 0}, "constant", 0).view({-1, 3, 1}); 
		torch::Tensor t0 = torch::zeros_like(matrix); 
		return torch::cat({t0, t0, matrix}, -1); 
	}
	
	const torch::Tensor H_Matrix(
			torch::Tensor Sols_, std::vector<torch::Tensor> b_C, 
			std::vector<torch::Tensor> mu_C, torch::Tensor mu_P, 
			torch::Tensor mu_phi) 
	{
		torch::Tensor H_ = _H_Matrix(Sols_, mu_P); 
		torch::Tensor theta_ = PhysicsCUDA::Theta(mu_C[0], mu_C[1], mu_C[2]);
		torch::Tensor Rz = OperatorsCUDA::Rz(-mu_phi); 
		torch::Tensor Ry = OperatorsCUDA::Ry(torch::acos(torch::zeros_like(mu_phi)) - theta_); 
		torch::Tensor Rx = OperatorsCUDA::Mul(Rz, torch::cat(b_C, -1).view({-1, 3, 1})); 	
		Rx = OperatorsCUDA::Mul(Ry, Rx.view({-1, 3, 1})); 
		Rx = -torch::atan2(Rx.index({torch::indexing::Slice(), 2}), Rx.index({torch::indexing::Slice(), 1})).view({-1, 1}); 
		Rx = OperatorsCUDA::Rx(Rx); 

		Rx = torch::transpose(Rx, 1, 2).contiguous(); 
		Ry = torch::transpose(Ry, 1, 2).contiguous(); 
		Rz = torch::transpose(Rz, 1, 2).contiguous(); 

		return OperatorsCUDA::Mul(OperatorsCUDA::Mul(Rz, OperatorsCUDA::Mul(Ry, Rx)), H_); 
	}

	const torch::Tensor Derivative(torch::Tensor x)
	{
		return OperatorsCUDA::Mul(OperatorsCUDA::Rz(_Pi_2(x)), _Unit(x, {1, 1, 0}));
	}

	const std::vector<torch::Tensor> Intersection(torch::Tensor A, torch::Tensor B)
	{

		torch::Tensor swp = torch::abs(OperatorsCUDA::Det(B)) > torch::abs(OperatorsCUDA::Det(A));
		torch::Tensor _tmp = B.index({swp}); 
		B.index_put_({swp}, A.index({swp})); 
		A.index_put_({swp}, _tmp);
		
		_tmp = OperatorsCUDA::Inv(A); 
		_tmp = OperatorsCUDA::Mul(_tmp, B);
		_tmp = torch::linalg::eigvals(_tmp); 
		torch::Tensor _r = torch::real(_tmp); 
		torch::Tensor msk = torch::isreal(_tmp)*torch::arange(3, 0, -1, torch::TensorOptions().device(A.device())); 
		msk = torch::argmax(msk, -1, true); 
		_r = torch::gather(_r, 1, msk).view({-1, 1, 1}); 
		torch::Tensor G = B - _r*A;
		torch::Tensor Q = _Factorization(G); 
		Q = _Factorization(G, Q, OperatorsCUDA::Cofactors(Q)); 
		Q = _SwapXY(G, Q); 
		torch::Tensor Lines = Q; 	
		
		Q = Q.view({-1, 3, 1, 3}); 	
		_tmp = torch::zeros_like(Q); 
		Q = torch::cat({Q, Q, Q}, 2).view({-1, 9, 1, 3}); 
		_tmp = torch::cat({_tmp, _tmp, _tmp}, 1); 
		Q = torch::cat({_tmp, Q}, 2); 
		
		_tmp = torch::cat({A.view({-1, 1, 3, 3}), A.view({-1, 1, 3, 3}), A.view({-1, 1, 3, 3})}, 1).view({-1, 9, 1, 3}); 
		Q = torch::cat({Q, _tmp}, 2); 
		Q = OperatorsCUDA::Cofactors(Q.view({-1, 3, 3})); 
		Q = Q.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).view({-1, 3, 3, 3}); 

		Q = std::get<1>(torch::linalg::eig(torch::transpose(Q, 2, 3))); 
		Q = torch::real(Q);
		Q = torch::transpose(Q, 2, 3); 

		return _EllipseLines(Lines, Q, A); 
		
		return {Q, Lines, _tmp, A}; 

		
	}
}

namespace SingleNuCUDA
{
	const torch::Tensor Sigma2(
			torch::Tensor Sxx, torch::Tensor Sxy, 
			torch::Tensor Syx, torch::Tensor Syy)
	{
		torch::Tensor _S = torch::cat({Sxx, Sxy, Syx, Syy}, -1).view({-1, 2, 2});
		_S = torch::inverse(_S); 
		_S = torch::pad(_S, {0, 1, 0, 1}, "constant", 0);
		_S = torch::transpose(_S, 1, 2);
		return _S.contiguous(); 
	}


	const std::vector<torch::Tensor> Nu(torch::Tensor b, torch::Tensor mu, 
			torch::Tensor met, torch::Tensor phi, 
			torch::Tensor Sxx, torch::Tensor Sxy, torch::Tensor Syx, torch::Tensor Syy,
			torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu)
	{


		// ---- Polar Version of Particles ---- //
		std::vector<torch::Tensor> b_P = NuSolCUDA::_Format(b.view({-1, 4}), 4);
		std::vector<torch::Tensor> mu_P = NuSolCUDA::_Format(mu.view({-1, 4}), 4); 
		
		// ---- Cartesian Version of Particles ---- //
		std::vector<torch::Tensor> b_C = NuSolCUDA::_Format(TransformCUDA::PxPyPz(b_P[0], b_P[1], b_P[2]), 3); 
		std::vector<torch::Tensor> mu_C = NuSolCUDA::_Format(TransformCUDA::PxPyPz(mu_P[0], mu_P[1], mu_P[2]), 3); 
		torch::Tensor muP_ = PhysicsCUDA::P(mu_C[0], mu_C[1], mu_C[2]); 

		// ---- Cartesian Version of Event Met ---- //
		NuSolCUDA::_CheckTensors({met, phi}); 
		torch::Tensor met_x = TransformCUDA::Px(met, phi); 
		torch::Tensor met_y = TransformCUDA::Py(met, phi);
		
		// ---- Precalculate the Mass Squared ---- //
		torch::Tensor mT2 = OperatorsCUDA::Dot(mT, mT); 
		torch::Tensor mW2 = OperatorsCUDA::Dot(mW, mW); 
		torch::Tensor mNu2 = OperatorsCUDA::Dot(mNu, mNu); 
		
		// ---- MET uncertainity ---- //	
		torch::Tensor S2_ = Sigma2(Sxx, Sxy, Syx, Syy);	
		
		torch::Tensor sols_ = NuSolCUDA::Solutions(b_P, b_C, mu_P, mu_C, mT2, mW2, mNu2);
		torch::Tensor H_ = NuSolCUDA::H_Matrix(sols_, b_C, mu_C, muP_, mu_P[2]); 
	
		torch::Tensor delta_ = NuSolCUDA::V0(met_x, met_y) - H_; 
		torch::Tensor X_ = OperatorsCUDA::Mul(torch::transpose(delta_, 1, 2).contiguous(), S2_); 
		X_ = OperatorsCUDA::Mul(X_, delta_).view({-1, 3, 3}); 
		
		torch::Tensor M_ = OperatorsCUDA::Mul(X_, NuSolCUDA::Derivative(X_)); 
		M_ = M_ + torch::transpose(M_, 1, 2); 
		return NuSolCUDA::Intersection(M_, _Unit(M_, {1, 1, -1}));
	}

}

namespace DoubleNuCUDA
{
	const torch::Tensor N(torch::Tensor H)
	{
		torch::Tensor H_ = torch::clone(H); 
		H_.index_put_({torch::indexing::Slice(), 2, torch::indexing::Slice()}, 0); 
		H_.index_put_({torch::indexing::Slice(), 2, 2}, 1);
		H_ = OperatorsCUDA::Inv(H_.contiguous());
		torch::Tensor H_T = torch::transpose(H_, 1, 2).contiguous(); 
		H_T = OperatorsCUDA::Mul(H_T, _Unit(H_, {1, 1, -1})); 
		return OperatorsCUDA::Mul(H_T, H_); 
	}

	const std::vector<torch::Tensor> NuNu(
			torch::Tensor b, torch::Tensor b_, torch::Tensor mu, torch::Tensor mu_, 
			torch::Tensor met, torch::Tensor phi, torch::Tensor mT, torch::Tensor mW, torch::Tensor mNu)
	{
		// ---- Polar Version of Particles ---- //
		std::vector<torch::Tensor> b_P = NuSolCUDA::_Format(b.view({-1, 4}), 4);
		std::vector<torch::Tensor> b__P = NuSolCUDA::_Format(b_.view({-1, 4}), 4);
		std::vector<torch::Tensor> mu_P = NuSolCUDA::_Format(mu.view({-1, 4}), 4); 
		std::vector<torch::Tensor> mu__P = NuSolCUDA::_Format(mu_.view({-1, 4}), 4); 
		
		// ---- Cartesian Version of Particles ---- //
		std::vector<torch::Tensor> b_C = NuSolCUDA::_Format(TransformCUDA::PxPyPz(b_P[0], b_P[1], b_P[2]), 3); 
		std::vector<torch::Tensor> b__C = NuSolCUDA::_Format(TransformCUDA::PxPyPz(b__P[0], b__P[1], b__P[2]), 3); 
		std::vector<torch::Tensor> mu_C = NuSolCUDA::_Format(TransformCUDA::PxPyPz(mu_P[0], mu_P[1], mu_P[2]), 3); 
		std::vector<torch::Tensor> mu__C = NuSolCUDA::_Format(TransformCUDA::PxPyPz(mu__P[0], mu__P[1], mu__P[2]), 3); 
		torch::Tensor muP_ = PhysicsCUDA::P(mu_C[0], mu_C[1], mu_C[2]); 
		torch::Tensor muP__ = PhysicsCUDA::P(mu__C[0], mu__C[1], mu__C[2]); 

		// ---- Cartesian Version of Event Met ---- //
		NuSolCUDA::_CheckTensors({met, phi}); 
		torch::Tensor met_x = TransformCUDA::Px(met, phi); 
		torch::Tensor met_y = TransformCUDA::Py(met, phi);
		
		// ---- Precalculate the Mass Squared ---- //
		torch::Tensor mT2 = OperatorsCUDA::Dot(mT, mT); 
		torch::Tensor mW2 = OperatorsCUDA::Dot(mW, mW); 
		torch::Tensor mNu2 = OperatorsCUDA::Dot(mNu, mNu); 

		// ---- Starting the algorithm ---- //
		torch::Tensor sols_ = NuSolCUDA::Solutions(b_P, b_C, mu_P, mu_C, mT2, mW2, mNu2);
		torch::Tensor sols__ = NuSolCUDA::Solutions(b__P, b__C, mu__P, mu__C, mT2, mW2, mNu2);
		
		torch::Tensor H_ = NuSolCUDA::H_Matrix(sols_, b_C, mu_C, muP_, mu_P[2]); 
		torch::Tensor H__ = NuSolCUDA::H_Matrix(sols__, b__C, mu__C, muP__, mu__P[2]); 


		// ---- Protection Against non-invertible Matrices ---- //
		torch::Tensor SkipEvent = OperatorsCUDA::Dot(OperatorsCUDA::Det(H_).view({-1, 1}), OperatorsCUDA::Det(H__).view({-1, 1})) != 0.; 
		SkipEvent = SkipEvent.view({-1}); 
		H_ = H_.index({SkipEvent}); 
		H__ = H__.index({SkipEvent}); 
		met_x = met_x.index({SkipEvent}); 
		met_y = met_y.index({SkipEvent});

		torch::Tensor N_ = DoubleNuCUDA::N(H_); 
		torch::Tensor N__ = DoubleNuCUDA::N(H__); 

		torch::Tensor S_ = NuSolCUDA::V0(met_x, met_y) - _Unit(met_y, {1, 1, -1});
		torch::Tensor n_ = OperatorsCUDA::Mul(OperatorsCUDA::Mul(S_.transpose(1, 2).contiguous(), N__), S_); 

		return NuSolCUDA::Intersection(N_, n_);
	}
}
#endif 
