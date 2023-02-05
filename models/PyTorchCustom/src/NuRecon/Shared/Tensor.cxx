#include "../Headers/NuSolTensor.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("Solutions", &NuSolTensors::Solutions, "Solutions"); 
	m.def("Nu", &SingleNuTensor::Nu, "Nu");


}
