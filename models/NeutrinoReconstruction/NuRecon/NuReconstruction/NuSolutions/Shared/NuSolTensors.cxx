#include <torch/extension.h>
#include "../Headers/NuSolTensors.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("x0Polar", &NuSolutionTensors::x0Polar, "x0Polar"); 
	m.def("x0Cartesian", &NuSolutionTensors::x0Cartesian, "x0Cartesian"); 
}
