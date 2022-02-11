#include <torch/extension.h>
#include <iostream>
#include <vector>

// CUDA forward declaration 
std::vector<torch::Tensor> ToCartesianCUDA(torch::Tensor eta, torch::Tensor phi, torch::Tensor pt, torch::Tensor e); 
torch::Tensor PathMassCartesianCUDA(torch::Tensor x, torch::Tensor y, torch::Tensor z, torch::Tensor e, torch::Tensor Combinations); 

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), "#x must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> ToCartesian(torch::Tensor eta, torch::Tensor phi, torch::Tensor pt, torch::Tensor e)
{

  CHECK_INPUT(eta); 
  CHECK_INPUT(phi);
  CHECK_INPUT(pt); 
  CHECK_INPUT(e); 

  return ToCartesianCUDA(eta, phi, pt, e);
}

torch::Tensor PathMassCartesian(torch::Tensor x, torch::Tensor y, torch::Tensor z, torch::Tensor e, torch::Tensor Combinations)
{

  CHECK_INPUT(x); 
  CHECK_INPUT(y);
  CHECK_INPUT(z); 
  CHECK_INPUT(e); 
  CHECK_INPUT(Combinations); 

  return PathMassCartesianCUDA(x, y, z, e, Combinations);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("ToCartesian", &ToCartesian, "Convert to Cartesian");
  m.def("ToCartesianCUDA", &ToCartesianCUDA, "Convert Detector to Cartesian (CUDA)");
  m.def("PathMassCartesian", &PathMassCartesian, "Convert Path to Mass"); 
  m.def("PathMassCartesianCUDA", &PathMassCartesianCUDA, "Convert Path to Mass (CUDA)"); 

}
