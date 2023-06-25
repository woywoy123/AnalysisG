#include "polar.h"

double Transform::Floats::Pt(double px, double py)
{
    return std::sqrt( std::pow(px, 2) + std::pow( py, 2) ); 
}

double Transform::Floats::Phi(double px, double py)
{
    return std::atan2(py, px); 
}

double Transform::Floats::PtEta(double pt, double pz)
{
    return std::asinh(pz / pt); 
}

double Transform::Floats::Eta(double px, double py, double pz)
{
    double pt = Transform::Floats::Pt(py, px); 
    return Transform::Floats::PtEta(pt, pz); 
}

std::vector<double> Transform::Floats::PtEtaPhi(double px, double py, double pz)
{
    double pt = Transform::Floats::Pt(px, py); 
    return {
            pt, 
            Transform::Floats::PtEta(pt, pz),
            Transform::Floats::Phi(px, py)
    }; 
}

