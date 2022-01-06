#include <torch/extension.h>
#include <iostream>
#include <vector>

// CUDA forward declaration 
std::vector<torch::Tensor> FastMassMultiplicationCUDA(torch::Tensor eta, torch::Tensor phi, torch::Tensor pt, torch::Tensor e, torch::Tensor Combinations); 

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), "#x must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> FastMassMultiplication(torch::Tensor eta, torch::Tensor phi, torch::Tensor pt, torch::Tensor e, torch::Tensor, torch::Tensor Combinations)
{

  CHECK_INPUT(eta); 
  CHECK_INPUT(phi);
  CHECK_INPUT(pt); 
  CHECK_INPUT(e); 
  CHECK_INPUT(Combinations);

  return FastMassMultiplicationCUDA(eta, phi, pt, e, Combinations);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("FastMassMultiplication", &FastMassMultiplication, "Fast MassMultiplication");
  m.def("FastMassMultiplicationCUDA", &FastMassMultiplicationCUDA, "Fast MassMultiplication (CUDA)");
}
