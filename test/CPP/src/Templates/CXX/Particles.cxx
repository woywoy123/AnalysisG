#include "../Headers/Particles.h"
#include "../Headers/Tools.h"

CyTemplate::CyParticle::CyParticle(){}
CyTemplate::CyParticle::CyParticle(double px, double py, double pz, double e)
{
    this -> _px = px; 
    this -> _py = py; 
    this -> _pz = pz; 
    this -> _e = e; 
}

CyTemplate::CyParticle::~CyParticle(){}

void CyTemplate::CyParticle::_UpdateState()
{
    if (!this -> _edited){ return; }
    this -> px(); 
    this -> py(); 
    this -> pz(); 
    this -> pt(); 
    this -> eta(); 
    this -> phi(); 
    this -> Mass(); 
    this -> e();
    this -> Hash(); 
    this -> _edited = false; 
}

std::string CyTemplate::CyParticle::Hash()
{
    if (this -> _edited)
    {
            this -> _hash = ""; 
            this -> _hash += Tools::ToString(this -> _px); 
            this -> _hash += Tools::ToString(this -> _py); 
            this -> _hash += Tools::ToString(this -> _pz); 
            this -> _hash += Tools::ToString(this -> _e); 
            this -> _hash = Tools::Hashing(this -> _hash); 
    }
    return this -> _hash; 
}

// Cartesian Conversion functions
double CyTemplate::CyParticle::px()
{
    if (this -> _edited && this -> _pt && this -> _phi)
    {
        this -> _px = (this -> _pt)*std::cos(this -> _phi);
    }
    return this -> _px; 
}

double CyTemplate::CyParticle::py()
{
    if (this -> _edited && this -> _pt && this -> _phi)
    {
        this -> _py = (this -> _pt)*std::sin(this -> _phi);
    }
    return this -> _py; 
}

double CyTemplate::CyParticle::pz()
{
    if (this -> _edited && this -> _pt && this -> _eta)
    {
        this -> _pz = (this -> _pt) * std::sinh(this -> _eta); 
    }
    return this -> _pz; 
}

// Polar Conversion functions
double CyTemplate::CyParticle::pt()
{
    if (this -> _edited && this -> _px && this -> _py)
    {
        double sum = 0; 
        sum += std::pow(this -> _px, 2); 
        sum += std::pow(this -> _py, 2);
        sum = std::pow(sum, 0.5); 
        this -> _pt = sum; 
    }
    return this -> _pt; 
}

double CyTemplate::CyParticle::eta()
{
    if (this -> _edited) { this -> pt(); this -> pz(); }
    if (this -> _edited && this -> _pz && this -> _pt)
    {
        this -> _eta = std::asinh(this -> _pz / this -> _pt); 
    }
    return this -> _eta; 
}

double CyTemplate::CyParticle::phi()
{
    if (this -> _edited && this -> _py && this -> _px)
    {
        this -> _phi = std::atan2(this -> _py, this -> _px); 
    }
    return this -> _phi; 
}

double CyTemplate::CyParticle::e()
{
    if (this -> _edited)
    {
        double sum = 0; 
        sum += std::pow(this -> Mass(), 2); 
        sum += std::pow(this -> px(), 2); 
        sum += std::pow(this -> py(), 2); 
        sum += std::pow(this -> pz(), 2); 
        this -> _e = std::pow(sum, 0.5); 
    }
    return this -> _e; 
}

// Cartesian Conversion functions
void CyTemplate::CyParticle::px(double val)
{ 
    this -> _px = val; 
    this -> _edited = true; 
}

void CyTemplate::CyParticle::py(double val)
{ 
    this -> _py = val; 
    this -> _edited = true; 
}

void CyTemplate::CyParticle::pz(double val)
{ 
    this -> _pz = val; 
    this -> _edited = true; 
}

// Polar Conversion functions
void CyTemplate::CyParticle::pt(double val)
{ 
    this -> _pt = val; 
    this -> _edited = true; 
}

void CyTemplate::CyParticle::eta(double val)
{ 
    this -> _eta = val; 
    this -> _edited = true; 
}

void CyTemplate::CyParticle::phi(double val)
{ 
    this -> _phi = val; 
    this -> _edited = true; 
}

void CyTemplate::CyParticle::e(double val)
{
    this -> _e = val; 
    this -> _edited = true; 
}

// Physics Functions 
double CyTemplate::CyParticle::Mass()
{
    double sum = 0; 
    sum += std::pow(this -> _e, 2); 
    sum -= std::pow(this -> px(), 2);
    sum -= std::pow(this -> py(), 2);
    sum -= std::pow(this -> pz(), 2);

    this -> _mass = (sum >= 0) ? std::pow(sum, 0.5): 0; 
    return this -> _mass; 
}

void CyTemplate::CyParticle::Mass(double val)
{
    this -> _mass = val; 
    this -> _edited = true; 
}

double CyTemplate::CyParticle::DeltaR(const CyTemplate::CyParticle& p)
{
    double sum = std::pow(this -> _eta - p._eta, 2); 
    sum += std::pow(std::atan(std::tan(this -> _eta)) - std::atan(std::tan(p._eta)), 2); 
    return std::pow(sum, 0.5); 
}
