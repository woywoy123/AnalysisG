#include "../Headers/PhysicsTensors.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("ToPxPyPzE", &PhysicsTensors::ToPxPyPzE, "ToPxPyPzE"); 

	m.def("Mass2Polar", &PhysicsTensors::Mass2Polar, "Mass2Polar");
	m.def("MassPolar", &PhysicsTensors::MassPolar, "MassPolar"); 

	m.def("Mass2Cartesian", &PhysicsTensors::Mass2Cartesian, "Mass2Cartesian"); 
	m.def("MassCartesian", &PhysicsTensors::MassCartesian, "MassCartesian"); 

	m.def("BetaPolar", &PhysicsTensors::BetaPolar, "BetaPolar"); 
	m.def("BetaCartesian", &PhysicsTensors::BetaCartesian, "BetaCartesian");

	m.def("CosThetaCartesian", &PhysicsTensors::CosThetaCartesian, "CosThetaCartesian"); 
	m.def("SinThetaCartesian", &PhysicsTensors::SinThetaCartesian, "SinThetaCartesian");
}


