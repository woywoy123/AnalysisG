#include "../Headers/CUDA.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("Dot", &OperatorsCUDA::Dot, "Dot Product");
	m.def("CosTheta", &OperatorsCUDA::CosTheta, "CosTheta");
	m.def("SinTheta", &OperatorsCUDA::SinTheta, "SinTheta");
}
