#include "NuSolKernel.cu"

torch::Tensor _Solutions(
		std::vector<torch::Tensor> b_P, std::vector<torch::Tensor> b_C, 
		std::vector<torch::Tensor> mu_P, std::vector<torch::Tensor> mu_C, 
		torch::Tensor mT2, torch::Tensor mW2, torch::Tensor mNu2)
{
	
	torch::Tensor c_ = torch::zeros_like(b_P[0]);  
	torch::Tensor s_ = torch::zeros_like(b_P[0]); 
	torch::Tensor x0p = torch::zeros_like(b_P[0]); 
	torch::Tensor x0 = torch::zeros_like(b_P[0]); 
		
	const int x = c_.size(0); 
	const int threads = 1024;	


	//const int x = c_size(0); 


	//const int x = v1.size(0); 
	//const int y = v1.size(1); 
	//const int threads = 1024; 
	//torch::Tensor _v12 = torch::zeros_like(v1);
	//torch::Tensor _v22 = torch::zeros_like(v1);
	//torch::Tensor _V1V2 = torch::zeros_like(v1);

	//const dim3 blocks((x + threads -1) / threads, y); 
	//const dim3 blocks2((x + threads -1) / threads); 
	//AT_DISPATCH_FLOATING_TYPES(v1.type(), "_Dot2K", ([&]
	//{
	//	_Dot2K<scalar_t><<<blocks, threads>>>(
	//			v1.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
	//			v1.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
	//			_v12.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
	//			x, y
	//	); 
	//	_Dot2K<scalar_t><<<blocks, threads>>>(
	//			v2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
	//			v2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
	//			_v22.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
	//			x, y
	//	); 
	//	_Dot2K<scalar_t><<<blocks, threads>>>(
	//			v1.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
	//			v2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
	//			_V1V2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
	//			x, y
	//	); 
	//})); 
	//_V1V2 = _V1V2.sum({-1}, true);
	//_v12 = _v12.sum({-1}, true);
	//_v22 = _v22.sum({-1}, true);
	//AT_DISPATCH_FLOATING_TYPES(v1.type(), "_CosThetaK", ([&]
	//{
	//	_CosThetaK<scalar_t><<<blocks2, threads>>>(
	//			_v12.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
	//			_v22.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
	//			_V1V2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
	//			x
	//	); 
	//})); 


	return c_; 
}
