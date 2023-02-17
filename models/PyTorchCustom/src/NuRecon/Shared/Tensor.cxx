#include "../Headers/NuSolTensor.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("Nu", &SingleNuTensor::Nu, "Nu");
	m.def("NuNu", &DoubleNuTensor::NuNu, "NuNu"); 
}
