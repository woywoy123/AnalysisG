#include "../Headers/PhysicsFloats.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("ToPx", &PhysicsFloats::ToPx, "ToPx"); 
	m.def("ToPy", &PhysicsFloats::ToPy, "ToPy"); 
	m.def("ToPz", &PhysicsFloats::ToPz, "ToPz"); 
	m.def("ToPxPyPzE", &PhysicsFloats::ToPxPyPzE, "ToPxPyPzE"); 
	
	m.def("Mass2Polar", &PhysicsFloats::Mass2Polar, "Mass2Polar");
	m.def("MassPolar", &PhysicsFloats::MassPolar, "MassPolar"); 
	m.def("Mass2Cartesian", &PhysicsFloats::Mass2Cartesian, "Mass2Cartesian"); 
	m.def("MassCartesian", &PhysicsFloats::MassCartesian, "MassCartesian"); 

	m.def("BetaPolar", &PhysicsFloats::BetaPolar, "BetaPolar"); 
	m.def("BetaCartesian", &PhysicsFloats::BetaCartesian, "BetaCartesian");

	m.def("CosThetaCartesian", &PhysicsFloats::CosThetaCartesian, "CosThetaCartesian"); 
	m.def("SinThetaCartesian", &PhysicsFloats::SinThetaCartesian, "SinThetaCartesian"); 
}


