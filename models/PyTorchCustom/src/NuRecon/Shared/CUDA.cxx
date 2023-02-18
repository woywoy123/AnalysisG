#include "../Headers/NuSolCUDA.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("NuPtEtaPhiE", &NuCUDA::NuPtEtaPhiE, "Nu");
	m.def("NuPxPyPzE", &NuCUDA::NuPxPyPzE, "Nu");
	m.def("NuDoublePtEtaPhiE", &NuCUDA::Nu_AsDouble_PtEtaPhiE, "Nu");
	m.def("NuDoublePxPyPzE", &NuCUDA::Nu_AsDouble_PxPyPzE, "Nu");
	m.def("NuListPtEtaPhiE", &NuCUDA::Nu_AsDoubleList_PtEtaPhiE, "Nu");
	m.def("NuListPxPyPzE", &NuCUDA::Nu_AsDoubleList_PxPyPzE, "Nu");

	m.def("NuNu", &DoubleNuCUDA::NuNu, "NuNu");
}
