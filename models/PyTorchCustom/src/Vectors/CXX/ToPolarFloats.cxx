#include "../Headers/ToPolarFloats.h"

double VectorFloats::PT(double px, double py)
{
	return std::sqrt( std::pow( px, 2 ) + std::pow( py, 2 ) );
}

double VectorFloats::Phi(double px, double py)
{
	return std::atan2( py, px ); 
}

double VectorFloats::_Eta(double pt, double pz)
{
	return std::asinh(pz / pt); 
}

double VectorFloats::Eta(double px, double py, double pz)
{
	return _Eta(PT(px, py), pz); 
}

std::vector<double> VectorFloats::PtEtaPhi(double px, double py, double pz)
{
	double pt = PT(px, py); 
	return {pt, _Eta(pt, pz), Phi(px, py)}; 
}
