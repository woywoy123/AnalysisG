#include "../Headers/NuSolTensor.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("x0", &NuSolutionTensors::x0, "x0"); 
}
