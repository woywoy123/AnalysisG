#include <torch/extension.h>
#include "../Headers/NuSolFloats.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("x0Polar", &NuSolutionFloats::x0Polar, "x0Polar"); 
	m.def("x0Cartesian", &NuSolutionFloats::x0Cartesian, "x0Cartesian");
}
