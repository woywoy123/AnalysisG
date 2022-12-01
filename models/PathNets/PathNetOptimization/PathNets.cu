#include <sys/types.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


template <typename scalar_t>
__device__ __forceinline__ void _pt_to_px(scalar_t* _px, const scalar_t* _pt, const scalar_t* _phi)
{
	(*_px) = (*_pt)*cos((*_phi)); 
}

template <typename scalar_t>
__device__ __forceinline__ void _pt_to_py(scalar_t* _py, const scalar_t* _pt, const scalar_t* _phi)
{
	(*_py) = (*_pt)*sin((*_phi)); 
}

template <typename scalar_t>
__device__ __forceinline__ void _pt_to_pz(scalar_t* _pz, const scalar_t* _pt, const scalar_t* _eta)
{
	(*_pz) = (*_pt)*sinh((*_eta)); 
}

template <typename scalar_t>
__global__ void PtEtaPhiEPxPyPzEKernel(
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> FourVec,
		torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output)
{
	
	const int indx = blockIdx.x*blockDim.x + threadIdx.x; 
	const int comp = blockIdx.y; 
	if (indx >= output.size(0) || comp >= output.size(1)){return;}
	
	if (comp == 0){ _pt_to_px(&(output[indx][0]), &(FourVec[indx][0]), &(FourVec[indx][2])); }
	else if (comp == 1){ _pt_to_py(&(output[indx][1]), &(FourVec[indx][0]), &(FourVec[indx][2])); }
	else if (comp == 2){ _pt_to_pz(&(output[indx][2]), &(FourVec[indx][0]), &(FourVec[indx][1])); }
	else if (comp == 3){ output[indx][3] = FourVec[indx][3]; }

}

torch::Tensor ToPxPyPzE_CUDA(torch::Tensor FourVector)
{
	const int l = FourVector.size(0);
	const int threads = 1024; 
	const dim3 blocks((l + threads-1)/threads, 4, 1); 
	torch::TensorOptions opt = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA); 
	
	FourVector = FourVector.to(opt); 
	torch::Tensor output = torch::zeros({l, 4}, opt);
	
	AT_DISPATCH_FLOATING_TYPES(torch::kFloat, "PtEtaPhiEPxPyPzEKernel", ([&]
	{
		PtEtaPhiEPxPyPzEKernel<scalar_t><<<blocks, threads>>>(
				FourVector.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
				output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
		);
	})); 

	return output; 
}



template <typename scalar_t>
__device__ __forceinline__ void _dR(scalar_t* _delR, 
		const scalar_t* _eta1, const scalar_t* _eta2, 
		const scalar_t* _phi1, const scalar_t* _phi2)
{
	(*_delR) = sqrt( pow( (*_phi1)  - (*_phi2), 2 ) + pow( (*_eta1)  - (*_eta2), 2 ) );
}

template <typename scalar_t>
__global__ void DeltaRKernel(
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> FourVec1,
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> FourVec2,
		torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output)
{
	const int indx = blockIdx.x*blockDim.x + threadIdx.x; 
	if (indx >= output.size(0)){ return; }
	_dR(&(output[indx][0]), 
		&(FourVec1[indx][1]), &(FourVec2[indx][1]), 
		&(FourVec1[indx][2]), &(FourVec2[indx][2]));
}

torch::Tensor DeltaR_CUDA(torch::Tensor FV1, torch::Tensor FV2)
{
	const int l = FV1.size(0);
	const int threads = 1024; 
	const dim3 blocks((l + threads-1)/threads); 
	torch::TensorOptions opt = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA); 
	
	FV1 = FV1.to(opt); 
	FV2 = FV2.to(opt); 
	torch::Tensor output = torch::zeros({l, 1}, opt);
	
	AT_DISPATCH_FLOATING_TYPES(torch::kFloat, "DeltaRKernel", ([&]
	{
		DeltaRKernel<scalar_t><<<blocks, threads>>>(
				FV1.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
				FV2.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
				output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
		);
	})); 

	return output; 
}




template <typename scalar_t>
__device__ __forceinline__ void _CartSumMass(scalar_t* m_out, 
		const scalar_t* px, const scalar_t* py, 
		const scalar_t* pz, const scalar_t* e)
{
	(*m_out) = pow((*e), 2) - pow((*px), 2) - pow((*py), 2) - pow((*pz), 2); 
	(*m_out) = sqrt(abs(*m_out)); 
}

template <typename scalar_t>
__global__ void _MassKernel(
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> FourVector,
		torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Mass)
{
	const int indx = blockIdx.x*blockDim.x + threadIdx.x;
	if (indx >= Mass.size(0)) { return; }

	_CartSumMass(&(Mass[indx][0]), 
			&(FourVector[indx][0]), &(FourVector[indx][1]), 
			&(FourVector[indx][2]), &(FourVector[indx][3])); 
}


torch::Tensor Mass_CUDA(torch::Tensor FourVector)
{
	const int mx = FourVector.size(0);
	const int threads = 1024; 
	const dim3 blocks((mx + threads-1)/threads, 1);
	
	torch::TensorOptions opt = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA); 
	torch::Tensor Mass = torch::zeros({mx, 1}, opt);

	AT_DISPATCH_FLOATING_TYPES(torch::kFloat, "_MassKernel", ([&]
	{
		_MassKernel<scalar_t><<<blocks, threads>>>(
				FourVector.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
				Mass.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>() 
		);
	}));
	return Mass;
}


template <typename scalar_t>
__global__ void _MergeKernel(
		const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> Aggregated, 
		torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> FourVec)
{
	const int indx = blockIdx.x*blockDim.x + threadIdx.x;
	const int indy = blockIdx.y;

	if (indx >= Aggregated.size(0)){ return; }
	for (unsigned int i = 0; i < Aggregated[indx].size(0); i++)
	{
		FourVec[indx][indy] += Aggregated[indx][i][indy];
	}
}

torch::Tensor Sum_CUDA(torch::Tensor CubeVector)
{		
	const int mx = CubeVector.size(0); 
	const int threads = 1024; 
	const dim3 blocks((mx + threads-1)/threads, 4);
	
	torch::TensorOptions opt = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA); 
	torch::Tensor FourVec = torch::zeros({mx, 4}, opt);

	AT_DISPATCH_FLOATING_TYPES(torch::kFloat, "_MergeKernel", ([&]
	{
		_MergeKernel<scalar_t><<<blocks, threads>>>(
				CubeVector.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				FourVec.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>() 
		);
	}));

	return FourVec;
}



template <typename scalar_t>
__global__ void _AggregateKernel(
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> IncomingEdge, 
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> EdgeSelect, 
		const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> NodeIndex, 
		torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output, 
		const bool ConvertCart)
{
	const int indx = blockIdx.x*blockDim.x + threadIdx.x;
	const int indy = blockIdx.y; 
	const int indz = blockIdx.z;
	const int index = indy*output.size(0) + indx;
	if (index >= NodeIndex.size(0) || indx >= output.size(0)){return;}
	
	scalar_t Node = NodeIndex[index][0]; 
	if (!EdgeSelect[index][0]){}
	else if (!ConvertCart){ output[Node][indx][indz] += IncomingEdge[index][indz]; }
	else if (indz == 0){ _pt_to_px(&(output[Node][indx][indz]), &(IncomingEdge[index][0]), &(IncomingEdge[index][2])); }
	else if (indz == 1){ _pt_to_py(&(output[Node][indx][indz]), &(IncomingEdge[index][0]), &(IncomingEdge[index][2])); }
	else if (indz == 2){ _pt_to_pz(&(output[Node][indx][indz]), &(IncomingEdge[index][0]), &(IncomingEdge[index][1])); }
	else if (indz == 3){ output[Node][indx][indz] = IncomingEdge[index][3]; }
}


torch::Tensor AggregateIncomingEdges_CUDA(torch::Tensor IncomingEdge, torch::Tensor NodeIndex, torch::Tensor EdgeSelect, bool ConvertCart)
{	
	const int threads = 1024; 
	int mx = torch::max(NodeIndex).item<int>()+1; 

	torch::TensorOptions opt = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA); 
	torch::Tensor output = torch::zeros({mx, mx, 4}, opt);

	torch::Tensor FourVec = torch::zeros({mx, 4}, opt); 
	torch::Tensor Mass = torch::zeros({mx, 1}, opt); 

	IncomingEdge = IncomingEdge.to(opt);
	NodeIndex = NodeIndex.to(opt); 
	EdgeSelect = EdgeSelect.to(opt); 
	
	const dim3 blocks((mx + threads-1)/threads, mx, 4); 	
	AT_DISPATCH_FLOATING_TYPES(torch::kFloat, "_AggregateKernel", ([&]
	{
		_AggregateKernel<scalar_t><<<blocks, threads>>>(
				IncomingEdge.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
				EdgeSelect.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
				NodeIndex.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
				output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
				ConvertCart
		);
	})); 
		
	output = Sum_CUDA(output); 
	return Mass_CUDA(output); 
}


