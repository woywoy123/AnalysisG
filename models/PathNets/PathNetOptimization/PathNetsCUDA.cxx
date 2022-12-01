#include <torch/extension.h>

// CUDA forward declaration 
torch::Tensor ToPxPyPzE_CUDA(torch::Tensor FourVector);
torch::Tensor DeltaR_CUDA(torch::Tensor FourVector1, torch::Tensor FourVector2);
torch::Tensor AggregateIncomingEdges_CUDA(torch::Tensor IncomingEdges, torch::Tensor NodeIndex, torch::Tensor EdgeSelect, bool ConvertCart);
torch::Tensor Mass_CUDA(torch::Tensor); 

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous") 
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor ToPxPyPzE(torch::Tensor FourVector)
{
	CHECK_INPUT(FourVector); 
	return ToPxPyPzE_CUDA(FourVector); 
}

torch::Tensor ToDeltaR(torch::Tensor FourVector1, torch::Tensor FourVector2)
{
	CHECK_INPUT(FourVector1); 
	CHECK_INPUT(FourVector2);
	return DeltaR_CUDA(FourVector1, FourVector2); 
}

torch::Tensor Mass(torch::Tensor FourVector)
{
	CHECK_INPUT(FourVector); 
	return Mass_CUDA(FourVector); 
}

torch::Tensor AggregateIncomingEdges(torch::Tensor IncomingEdges, torch::Tensor NodeIndex, torch::Tensor EdgeSelect, bool ConvertCart)
{
	CHECK_INPUT(IncomingEdges); 
	CHECK_INPUT(NodeIndex); 
	CHECK_INPUT(EdgeSelect); 

	return AggregateIncomingEdges_CUDA(IncomingEdges, NodeIndex, EdgeSelect, ConvertCart);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("ToPxPyPzE", &ToPxPyPzE, "To ToPxPyPzE");
	m.def("ToDeltaR", &ToDeltaR, "To DeltaR");
	m.def("Mass", &Mass, "Calculate Mass"); 
	m.def("AggregateIncomingEdges", &AggregateIncomingEdges, "AggregateIncomingEdges"); 
}


