#include "../Headers/ToCartesianFloats.h" 

double VectorFloats::Px(double pt, double phi)
{
	return pt * std::cos(phi); 
}

double VectorFloats::Py(double pt, double phi)
{
	return pt * std::sin(phi); 
}

double VectorFloats::Pz(double pt, double eta)
{
	return pt * std::sinh(eta); 
}

std::vector<double> VectorFloats::PxPyPz(double pt, double eta, double phi)
{
	return {Px(pt, phi), Py(pt, phi), Pz(pt, eta)}; 
}
