#include "../Headers/ToCartesianCUDA.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("Px", &VectorCUDA::Px, "Px"); 
	m.def("Py", &VectorCUDA::Py, "Py"); 
	m.def("Pz", &VectorCUDA::Pz, "Pz"); 
	m.def("PxPyPz", &VectorCUDA::PxPyPz, "PxPyPz");
}
