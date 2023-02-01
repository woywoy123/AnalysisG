#include "../Headers/ToCartesianFloats.h"
#include "../Headers/ToPolarFloats.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("Px", &VectorFloats::Px, "Px"); 
	m.def("Py", &VectorFloats::Py, "Py"); 
	m.def("Pz", &VectorFloats::Pz, "Pz");
	m.def("PxPyPz", &VectorFloats::PxPyPz, "PxPyPz"); 

	m.def("PT", &VectorFloats::PT, "PT");
	m.def("Phi", &VectorFloats::Phi, "Phi");
	m.def("Eta", &VectorFloats::Eta, "Eta");
	m.def("PtEtaPhi", &VectorFloats::PtEtaPhi, "PtEtaPhi"); 
}
