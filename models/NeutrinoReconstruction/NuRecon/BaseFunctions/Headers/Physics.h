#ifndef H_PHYSICS
#define H_PHYSICS

#include <torch/extension.h>
#include <iostream>

torch::Tensor ToPx(float pt, float phi, std::string device = "cpu"); 
torch::Tensor ToPy(float pt, float phi, std::string device = "cpu"); 
torch::Tensor ToPz(float pt, float eta, std::string device = "cpu"); 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("ToPx", &ToPx, "ToPx"); 
	m.def("ToPy", &ToPy, "ToPy"); 
	m.def("ToPz", &ToPz, "ToPz"); 
}



#endif
