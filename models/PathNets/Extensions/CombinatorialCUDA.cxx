#include <torch/extension.h>
#include <vector>

using namespace torch::indexing; 

torch::Tensor MassFromPxPyPzE(torch::Tensor v)
{
  v = v.pow(2);  
  v = v.view({-1, 4});
  
  torch::Tensor px = v.index({Slice(), Slice(0, 1, 1)}); 
  torch::Tensor py = v.index({Slice(), Slice(1, 2, 2)}); 
  torch::Tensor pz = v.index({Slice(), Slice(2, 3, 3)}); 
  torch::Tensor e = v.index({Slice(), Slice(3, 4, 4)}); 
  
  torch::Tensor s2 = e - px - py - pz;
  return torch::sqrt(s2.abs()); 
}

// CUDA forward declaration
torch::Tensor PathVectorGPU(torch::Tensor AdjMatrix, torch::Tensor FourVector); 
std::vector<torch::Tensor> IncomingEdgeVectorGPU(torch::Tensor AdjMatrix, torch::Tensor IncomingEdges, torch::Tensor Index);
//torch::Tensor PathCombinatorialGPU(const int n, torch::Tensor t);
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous") 
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor PathVectorCUDA(torch::Tensor AdjMatrix, torch::Tensor FourVector)
{
	CHECK_INPUT(AdjMatrix); 
	CHECK_INPUT(FourVector);
	
	return PathVectorGPU(AdjMatrix, FourVector);
}


torch::Tensor PathMassCUDA(torch::Tensor AdjMatrix, torch::Tensor FourVector)
{
	return MassFromPxPyPzE(PathVectorCUDA(AdjMatrix, FourVector)); 
}


std::vector<torch::Tensor> IncomingEdgeVectorCUDA(torch::Tensor AdjMatrix, torch::Tensor IncomingEdges, torch::Tensor Index)
{
	CHECK_INPUT(AdjMatrix);
	CHECK_INPUT(IncomingEdges);
	CHECK_INPUT(Index);
		
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kLong); 
	std::vector<torch::Tensor> V = IncomingEdgeVectorGPU(AdjMatrix, IncomingEdges, Index);
	return {V[0], V[1].to(options)};
}

std::vector<torch::Tensor> IncomingEdgeMassCUDA(torch::Tensor AdjMatrix, torch::Tensor IncomingEdges, torch::Tensor Index)
{
	CHECK_INPUT(AdjMatrix);
	CHECK_INPUT(IncomingEdges);
	CHECK_INPUT(Index);
	
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kLong); 
	std::vector<torch::Tensor> V = IncomingEdgeVectorGPU(AdjMatrix, IncomingEdges, Index);
	return {MassFromPxPyPzE(V[0]), V[1].to(options)};
}

//torch::Tensor PathCombinatorialCUDA(int n, int k)
//{
//	std::vector<torch::Tensor> out;
//	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
//	for (int i = 1; i < k+1; i++){Combinatorial(n, i, 0, &out, options);}
//	torch::Tensor t = torch::stack(out);
//	CHECK_INPUT(t);
//
//	return PathCombinatorialGPU(n, t);
//}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("PathVectorCUDA", &PathVectorCUDA, "Summation of four vectors");
	m.def("PathMassCUDA", &PathMassCUDA, "Invariant Mass"); 
	m.def("IncomingEdgeVectorCUDA", &IncomingEdgeVectorCUDA, "Computes the aggregated vector for different combinatorial of incoming edges");
	m.def("IncomingEdgeMassCUDA", &IncomingEdgeMassCUDA, "Computes the invariant mass of summed edge combinatorials");
}
