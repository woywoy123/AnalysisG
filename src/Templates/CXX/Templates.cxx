#include "../Headers/Templates.h"

CyTemplate::CyEventTemplate::CyEventTemplate(){}
CyTemplate::CyEventTemplate::~CyEventTemplate(){}

std::string CyTemplate::CyEventTemplate::Hash(){ return this -> _hash; }
void CyTemplate::CyEventTemplate::Hash(std::string inpt)
{
    if (this -> _hash != ""){return;}
    if (inpt.size() == 18){ this -> _hash = inpt; return; }
    inpt = inpt + "/" + Tools::ToString(this -> index) + "/" + (this -> tree); 
    this -> _hash = Tools::Hashing(inpt); 
}


CyTemplate::CyParticleTemplate::CyParticleTemplate(){}
CyTemplate::CyParticleTemplate::CyParticleTemplate(double px, double py, double pz, double e)
{
    this -> _px = px; 
    this -> _py = py; 
    this -> _pz = pz; 
    this -> _e = e; 
}
CyTemplate::CyParticleTemplate::~CyParticleTemplate(){}

void CyTemplate::CyParticleTemplate::_UpdateState()
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

std::string CyTemplate::CyParticleTemplate::Hash()
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
double CyTemplate::CyParticleTemplate::px()
{
    if (this -> _edited && this -> _pt && this -> _phi)
    {
        this -> _px = (this -> _pt)*std::cos(this -> _phi);
    }
    return this -> _px; 
}

double CyTemplate::CyParticleTemplate::py()
{
    if (this -> _edited && this -> _pt && this -> _phi)
    {
        this -> _py = (this -> _pt)*std::sin(this -> _phi);
    }
    return this -> _py; 
}

double CyTemplate::CyParticleTemplate::pz()
{
    if (this -> _edited && this -> _pt && this -> _eta)
    {
        this -> _pz = (this -> _pt) * std::sinh(this -> _eta); 
    }
    return this -> _pz; 
}

// Polar Conversion functions
double CyTemplate::CyParticleTemplate::pt()
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

double CyTemplate::CyParticleTemplate::eta()
{
    if (this -> _edited) { this -> pt(); this -> pz(); }
    if (this -> _edited && this -> _pz && this -> _pt)
    {
        this -> _eta = std::asinh(this -> _pz / this -> _pt); 
    }
    return this -> _eta; 
}

double CyTemplate::CyParticleTemplate::phi()
{
    if (this -> _edited && this -> _py && this -> _px)
    {
        this -> _phi = std::atan2(this -> _py, this -> _px); 
    }
    return this -> _phi; 
}

double CyTemplate::CyParticleTemplate::e()
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
void CyTemplate::CyParticleTemplate::px(double val)
{ 
    this -> _px = val; 
    this -> _edited = true; 
}

void CyTemplate::CyParticleTemplate::py(double val)
{ 
    this -> _py = val; 
    this -> _edited = true; 
}

void CyTemplate::CyParticleTemplate::pz(double val)
{ 
    this -> _pz = val; 
    this -> _edited = true; 
}

// Polar Conversion functions
void CyTemplate::CyParticleTemplate::pt(double val)
{ 
    this -> _pt = val; 
    this -> _edited = true; 
}

void CyTemplate::CyParticleTemplate::eta(double val)
{ 
    this -> _eta = val; 
    this -> _edited = true; 
}

void CyTemplate::CyParticleTemplate::phi(double val)
{ 
    this -> _phi = val; 
    this -> _edited = true; 
}

void CyTemplate::CyParticleTemplate::e(double val)
{
    this -> _e = val; 
    this -> _edited = true; 
}

// Physics Functions 
double CyTemplate::CyParticleTemplate::Mass()
{
    double sum = 0; 
    sum += std::pow(this -> _e, 2); 
    sum -= std::pow(this -> px(), 2);
    sum -= std::pow(this -> py(), 2);
    sum -= std::pow(this -> pz(), 2);

    this -> _mass = (sum >= 0) ? std::pow(sum, 0.5): 0; 
    return this -> _mass; 
}

void CyTemplate::CyParticleTemplate::Mass(double val)
{
    this -> _mass = val; 
    this -> _edited = true; 
}

double CyTemplate::CyParticleTemplate::DeltaR(const CyTemplate::CyParticleTemplate& p)
{
    double sum = std::pow(this -> _eta - p._eta, 2); 
    sum += std::pow(std::atan(std::tan(this -> _phi)) - std::atan(std::tan(p._phi)), 2); 
    return std::pow(sum, 0.5); 
}

// Particle Functions 
signed int CyTemplate::CyParticleTemplate::pdgid()
{
    return this -> _pdgid; 
}

void CyTemplate::CyParticleTemplate::pdgid(signed int id)
{
    this -> _pdgid = id; 
}

double CyTemplate::CyParticleTemplate::charge()
{
    return this -> _charge; 
}

void CyTemplate::CyParticleTemplate::charge(double id)
{
    this -> _charge = id; 
}

std::string CyTemplate::CyParticleTemplate::symbol()
{
    std::map<int, std::string> sym = {
                 {1, "d"}, {2, "u"}, {3, "s"}, {4, "c"}, {5, "b"}, {6, "t"},
                 {11, "e"}, {12, "nu_e"}, {13, "mu"}, {14, "nu_mu"}, {15, "tau"}, {16, "nu_{tau}"},
                 {21, "g"}, {22, "gamma"}
    }; 

    std::stringstream ss; 
    ss << sym[std::abs(this -> _pdgid)];
    return ss.str(); 
}

void CyTemplate::CyParticleTemplate::symbol(std::string inpt)
{
    this -> _symbol = inpt; 
}

bool CyTemplate::CyParticleTemplate::is(std::vector<signed int> p)
{
    for (signed int i : p){ if (std::abs(i) == std::abs(this -> _pdgid)) { return true; }}
    return false; 
}

bool CyTemplate::CyParticleTemplate::is_b(){ return this -> is({5}); }
bool CyTemplate::CyParticleTemplate::is_nu(){ return this -> is(this -> _nudef); }
bool CyTemplate::CyParticleTemplate::is_lep(){ return this -> is(this -> _lepdef); }
