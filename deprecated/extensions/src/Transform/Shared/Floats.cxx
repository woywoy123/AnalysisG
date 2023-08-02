#include "../Headers/ToCartesianFloats.h"
#include "../Headers/ToPolarFloats.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	// Basic Conversion to Cartesian from Polar 
	m.def("Px", &TransformFloats::Px, "Px"); 
	m.def("Py", &TransformFloats::Py, "Py"); 
	m.def("Pz", &TransformFloats::Pz, "Pz");
	m.def("PxPyPz", &TransformFloats::PxPyPz, "PxPyPz");
	
	// Basic Conversion to Polar 
	m.def("PT", &TransformFloats::PT, "PT");
	m.def("Phi", &TransformFloats::Phi, "Phi");
	m.def("Eta", &TransformFloats::Eta, "Eta");
	m.def("PtEtaPhi", &TransformFloats::PtEtaPhi, "PtEtaPhi");
}
