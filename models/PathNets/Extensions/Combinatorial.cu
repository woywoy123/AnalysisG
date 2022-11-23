#include <sys/types.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename scalar_t>
__device__ __forceinline__ void AggregatePath(
		scalar_t* Output, 
		const scalar_t* FourVec, 
		const scalar_t* Selector)
{
	(*Output) += (*FourVec)*(*Selector);  
}

template <typename scalar_t>
__global__ void SelectorKernel(
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> PTH, 
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> FV, 
	torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Pmu,
	const size_t cmbi_l, 
	const size_t nodes)
{
	const int nd = blockIdx.y;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < cmbi_l)
	{
		for (unsigned int i = 0; i < nodes; i++)
		{
			AggregatePath(
				&(Pmu[index][nd]), 
				&(FV[i][nd]), 
				&(PTH[index][i])); 
		}
	}
}

torch::Tensor PathVectorGPU(torch::Tensor AdjMatrix, torch::Tensor FourVector)
{
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
	const int cmbi_l = AdjMatrix.size(0);
	const int nodes = AdjMatrix.size(1);
	const int threads = 1024;
	const dim3 blocks((cmbi_l + threads -1) / threads, 4);
	
	torch::Tensor Pmu = torch::zeros({cmbi_l, 4}, options);
	FourVector = FourVector.to(options);
	AdjMatrix = AdjMatrix.to(options);

	AT_DISPATCH_FLOATING_TYPES(torch::kFloat, "SelectorKernel", ([&]
	{
		SelectorKernel<scalar_t><<<blocks, threads>>>(
			AdjMatrix.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
			FourVector.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			Pmu.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			cmbi_l, 
			nodes
		);
	}));
	return Pmu;
}



template <typename scalar_t>
__device__ __forceinline__ void AggregatePath(
		scalar_t* Output, 
		scalar_t* IncomingEdgeAdj, 
		const scalar_t* IncomingEdgeVector, 
		const scalar_t* AdjMat, 
		const scalar_t* edgeindex,
		const int* node, 
		const int* adj)
{
	if ((*node) != (*edgeindex) || (*AdjMat) == 0){return;}
	(*Output) += (*IncomingEdgeVector); 
	(*IncomingEdgeAdj) = (*adj);
}


template <typename scalar_t>
__global__ void NodeSelectorKernel(
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> PTH, 
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> FV, 
	const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> NodeIndex, 
	torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Pmu_adj,
	torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Pmu,
	const size_t cmbi_l, 
	const size_t nodes)
{
	const int adj = blockIdx.x*blockDim.x + threadIdx.x;
	const int nod = blockIdx.y;
	const int dim = blockIdx.z;
	const int index = adj + nod*PTH.size(0);

	if (index < cmbi_l && adj < PTH.size(0))
	{
		for (unsigned int i = 0; i < nodes; i++)
		{
			AggregatePath(
				&(Pmu[index][dim]),
				&(Pmu_adj[index][0]),
				&(FV[i+nod*nodes][dim]), 
				&(PTH[adj][i]), 
				&(NodeIndex[i + nod*nodes][0]), 
				&nod,
				&adj); 
		}
	}
}

std::vector<torch::Tensor> IncomingEdgeVectorGPU(torch::Tensor AdjMatrix, torch::Tensor IncomingEdges, torch::Tensor Index)
{

	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA); 
	const int adj = AdjMatrix.size(0);
	const int nodes = AdjMatrix.size(1);
	const int threads = 1024;
	const dim3 blocks((adj+threads-1)/threads, nodes, 4);

	torch::Tensor Pmu = torch::zeros({adj*nodes, 4}, options); 
	torch::Tensor Pmu_adj = torch::zeros({adj*nodes, 1}, options); 
	Index = Index.to(options);
	IncomingEdges = IncomingEdges.to(options);
	AdjMatrix = AdjMatrix.to(options);

	AT_DISPATCH_FLOATING_TYPES(torch::kFloat, "NodeSelectorKernel", ([&]
	{
		NodeSelectorKernel<scalar_t><<<blocks, threads>>>(
			AdjMatrix.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
			IncomingEdges.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			Index.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			Pmu_adj.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			Pmu.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
			Pmu.size(0), 
			nodes
		);
	})); 

	return {Pmu, Pmu_adj}; 
}
