#include "OperatorsKernel.cu"

torch::Tensor _Dot(torch::Tensor v1, torch::Tensor v2)
{
	const int l = v1.size(0);
	const int d = v1.size(1); 
	const int threads = 1024; 
	torch::Tensor _out = torch::zeros_like(v1); 
	
	const dim3 blocks((l + threads -1) / threads, d); 
	AT_DISPATCH_FLOATING_TYPES(v1.scalar_type(), "_Dot2K", ([&]
	{
		_Dot2K<scalar_t><<<blocks, threads>>>(
				v1.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				v2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				l, d
		); 
	})); 

	return _out; 
}

torch::Tensor _Mul(torch::Tensor v1, torch::Tensor v2)
{
	const int x = v1.size(0); 
	const int y = v1.size(1); 
	
	const int z1 = v1.size(2);
	const int z2 = v2.size(2);
	const int threads = 1024; 
	const dim3 blocks((x + threads -1) / threads, y, z1); 
	const dim3 blocks2((x + threads -1) / threads, y); 

	torch::Tensor _out = torch::zeros(
			{x, y, 1}, 
			torch::TensorOptions().dtype(v1.scalar_type()).device(v1.device()));
	

	AT_DISPATCH_FLOATING_TYPES(v1.scalar_type(), "_Dot3K", ([&]
	{
		_Dot3K<scalar_t><<<blocks, threads>>>(
				v1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
				v2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
				v1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
				x, y, z1, z2
		);
		_Sum3K<scalar_t><<<blocks2, threads>>>(
				v1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
				_out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
				x, y, z1
		); 
	})); 

	return _out; 
}



torch::Tensor _CosTheta(torch::Tensor v1, torch::Tensor v2)
{
	const int x = v1.size(0); 
	const int y = v1.size(1); 
	const int threads = 1024; 
	torch::Tensor _v12 = torch::zeros_like(v1);
	torch::Tensor _v22 = torch::zeros_like(v1);
	torch::Tensor _V1V2 = torch::zeros_like(v1);

	const dim3 blocks((x + threads -1) / threads, y); 
	const dim3 blocks2((x + threads -1) / threads); 
	AT_DISPATCH_FLOATING_TYPES(v1.scalar_type(), "_Dot2K", ([&]
	{
		_Dot2K<scalar_t><<<blocks, threads>>>(
				v1.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				v1.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_v12.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				x, y
		); 
		_Dot2K<scalar_t><<<blocks, threads>>>(
				v2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				v2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_v22.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				x, y
		); 
		_Dot2K<scalar_t><<<blocks, threads>>>(
				v1.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				v2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_V1V2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				x, y
		); 
	})); 
	_V1V2 = _V1V2.sum({-1}, true);
	_v12 = _v12.sum({-1}, true);
	_v22 = _v22.sum({-1}, true);
	AT_DISPATCH_FLOATING_TYPES(v1.scalar_type(), "_CosThetaK", ([&]
	{
		_CosThetaK<scalar_t><<<blocks2, threads>>>(
				_v12.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
				_v22.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
				_V1V2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
				x
		); 
	})); 
	return _V1V2; 
}

torch::Tensor _Rx(torch::Tensor angle)
{
	const int threads = 1024; 
	const int x = angle.size(0); 
	torch::TensorOptions op = torch::TensorOptions().dtype( angle.scalar_type() ).device( angle.device() ); 
	torch::Tensor _rx = torch::zeros({x, 3, 3}, op); 
	
	const dim3 blocks((x + threads -1) / threads, 3, 3); 
	AT_DISPATCH_FLOATING_TYPES(angle.scalar_type(), "_RxK", ([&]
	{
		_RxK<scalar_t><<<blocks, threads>>>(
				angle.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_rx.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()
		); 
	})); 
	return _rx; 
}

torch::Tensor _Ry(torch::Tensor angle)
{
	const int threads = 1024; 
	const int x = angle.size(0); 
	torch::TensorOptions op = torch::TensorOptions().dtype( angle.scalar_type() ).device( angle.device() ); 
	torch::Tensor _ry = torch::zeros({x, 3, 3}, op); 
	
	const dim3 blocks((x + threads -1) / threads, 3, 3); 
	AT_DISPATCH_FLOATING_TYPES(angle.scalar_type(), "_RyK", ([&]
	{
		_RyK<scalar_t><<<blocks, threads>>>(
				angle.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_ry.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()
		); 
	})); 
	return _ry; 
}

torch::Tensor _Rz(torch::Tensor angle)
{
	const int threads = 1024; 
	const int x = angle.size(0); 
	torch::TensorOptions op = torch::TensorOptions().dtype( angle.scalar_type() ).device( angle.device() ); 
	torch::Tensor _rz = torch::zeros({x, 3, 3}, op); 
	
	const dim3 blocks((x + threads -1) / threads, 3, 3); 
	AT_DISPATCH_FLOATING_TYPES(angle.scalar_type(), "_RzK", ([&]
	{
		_RzK<scalar_t><<<blocks, threads>>>(
				angle.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
				_rz.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()
		); 
	})); 
	return _rz; 
}

