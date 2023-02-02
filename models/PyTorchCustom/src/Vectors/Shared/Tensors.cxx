#include "../Headers/ToCartesianTensors.h"
#include "../Headers/ToPolarTensors.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("Px", &VectorTensors::Px, "Px"); 
	m.def("Py", &VectorTensors::Py, "Py"); 
	m.def("Pz", &VectorTensors::Pz, "Pz"); 
	m.def("PxPyPz", &VectorTensors::PxPyPz, "PxPyPz");

	m.def("PT",  &VectorTensors::PT, "PT");
	m.def("Phi", &VectorTensors::Phi, "Phi");
	m.def("Eta", &VectorTensors::Eta, "Eta");
	m.def("PtEtaPhi", &VectorTensors::PtEtaPhi, "PtEtaPhi"); 
}
