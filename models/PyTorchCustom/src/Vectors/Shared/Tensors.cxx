#include "../Headers/ToCartesianTensors.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("Px", &VectorTensors::Px, "Px"); 
	m.def("Py", &VectorTensors::Py, "Py"); 
	m.def("Pz", &VectorTensors::Pz, "Pz"); 
	m.def("PxPyPz", &VectorTensors::PxPyPz, "PxPyPz");
}
