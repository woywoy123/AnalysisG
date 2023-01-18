#include <torch/extension.h>
#include "../Headers/NuSolFloat.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("x0", &NuSolutionFloats::x0, "x0"); 
}
