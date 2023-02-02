#include "../Headers/ToCartesianCUDA.h"
#include "../Headers/ToPolarCUDA.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("Px", &VectorCUDA::Px, "Px"); 
	m.def("Py", &VectorCUDA::Py, "Py"); 
	m.def("Pz", &VectorCUDA::Pz, "Pz"); 
	m.def("PxPyPz", &VectorCUDA::PxPyPz, "PxPyPz");

	m.def("PT",  &VectorCUDA::PT, "PT");
	m.def("Phi", &VectorCUDA::Phi, "Phi");
	m.def("Eta", &VectorCUDA::Eta, "Eta");
	m.def("PtEtaPhi", &VectorCUDA::PtEtaPhi, "PtEtaPhi"); 

}
