#include <torch/extension.h>
#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z)
{
  auto s = torch::sigmoid(z); 
  return (1-s)*s;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("d_sigmoid", &d_sigmoid, "Path Sigmoid");
}
