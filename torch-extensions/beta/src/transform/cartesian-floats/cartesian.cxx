#include "cartesian.h"

double Transform::Floats::Px(double pt, double phi)
{
    return pt*std::cos(phi); 
}

double Transform::Floats::Py(double pt, double phi)
{
    return pt*std::sin(phi); 
}

double Transform::Floats::Pz(double pt, double eta)
{
    return pt*std::sinh(eta); 
}

std::vector<double> Transform::Floats::PxPyPz(double pt, double eta, double phi)
{
    return {Px(pt, phi), Py(pt, phi), Pz(pt, eta)}; 
}
