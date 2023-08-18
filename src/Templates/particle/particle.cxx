#include "particle.h"
#include "../tools/tools.h"
#include <cmath>

namespace CyTemplate
{
    CyParticleTemplate::CyParticleTemplate(){}
    CyParticleTemplate::CyParticleTemplate(
            double px, double py, double pz, double e) 
    {
        this -> _px = px; 
        this -> _py = py; 
        this -> _pz = pz; 
        this -> _e = e; 
        this -> _polar = true; 
    }

    CyParticleTemplate::CyParticleTemplate(
            double px, double py, double pz)
    {
        this -> _px = px; 
        this -> _py = py; 
        this -> _pz = pz; 
        this -> e(); 
        this -> _polar = true; 
    }

    CyParticleTemplate::~CyParticleTemplate(){}

    ExportParticleTemplate CyParticleTemplate::MakeMapping()
    {
        ExportParticleTemplate tmp; 
        tmp.e = this -> e(); 

        tmp.px = this -> px();
        tmp.py = this -> py(); 
        tmp.pz = this -> pz(); 

        tmp.pt = this -> pt(); 
        tmp.eta = this -> eta(); 
        tmp.phi = this -> phi(); 

        tmp.mass = this -> mass(); 
        tmp.charge = this -> charge(); 
        
        tmp.pdgid = this -> pdgid(); 
        tmp.index = this -> index; 
        
        tmp.hash = this -> hash(); 
        tmp.symbol = this -> symbol();
        
        tmp.lepdef = this -> lepdef; 
        tmp.nudef = this -> nudef; 
        return tmp; 
    }


    double CyParticleTemplate::e()
    {
        if (this -> _e >= 0){return this -> _e;}
        this -> _e += std::pow(this -> px(), 2); 
        this -> _e += std::pow(this -> py(), 2); 
        this -> _e += std::pow(this -> pz(), 2); 
        this -> _e  = std::pow(this -> _e, 0.5); 
        return this -> _e; 
    }

    void CyParticleTemplate::e(double val)
    { 
        this -> _e = val; 
    }

    void CyParticleTemplate::mass(double val)
    { 
        this -> _mass = val; 
    }

    double CyParticleTemplate::mass()
    {
        if (this -> _mass > -1){ return this -> _mass; }
        this -> _mass = 0; 
        this -> _mass -= std::pow(this -> px(), 2); 
        this -> _mass -= std::pow(this -> py(), 2); 
        this -> _mass -= std::pow(this -> pz(), 2); 
        this -> _mass += std::pow(this -> e() , 2); 
        this -> _mass = (this -> _mass >= 0) ? std::pow(this -> _mass, 0.5) : -1; 
        return this -> _mass; 
    }

    void CyParticleTemplate::symbol(std::string val)
    {
        this -> _symbol = val;
    }

    std::string CyParticleTemplate::symbol()
    {
        if ((this -> _symbol).size() != 0)
        { 
            return this -> _symbol; 
        }
        
        std::map<int, std::string> sym = {
                 {1, "d"}, {2, "u"}, {3, "s"}, 
                 {4, "c"}, {5, "b"}, {6, "t"},
                 {11, "e"}, {12, "$\\nu_{e}$"}, 
                 {13, "$\\mu$"}, {14, "$\\nu_{\\mu}$"}, 
                 {15, "$\\tau$"}, {16, "$\\nu_{\\tau}$"},
                 {21, "g"}, {22, "$\\gamma$"}
        }; 

        std::stringstream ss; 
        ss << sym[std::abs(this -> _pdgid)];
        return ss.str(); 
    }


    void CyParticleTemplate::pdgid(int val)
    {
        this -> _pdgid = val; 
    }

    int CyParticleTemplate::pdgid()
    {
        if (this -> _pdgid != 0){ return this -> _pdgid; }
        if ((this -> _symbol).size() == 0){ return this -> _pdgid; }
        
        std::map<int, std::string> sym = {
                 {1, "d"}, {2, "u"}, {3, "s"}, 
                 {4, "c"}, {5, "b"}, {6, "t"},
                 {11, "e"}, {12, "$\\nu_{e}$"}, 
                 {13, "$\\mu$"}, {14, "$\\nu_{\\mu}$"}, 
                 {15, "$\\tau$"}, {16, "$\\nu_{\\tau}$"},
                 {21, "g"}, {22, "$\\gamma$"}
        }; 
        std::map<int, std::string>::iterator it; 
        for (it = sym.begin(); it != sym.end(); ++it)
        {
            if (it -> second != this -> _symbol){continue;}
            this -> _pdgid = it -> first; 
            break; 
        }
        return this -> _pdgid;  
    }

    void CyParticleTemplate::charge(double val)
    {
        this -> _charge = val;
    }

    double CyParticleTemplate::charge()
    {
        return this -> _charge;
    }

    bool CyParticleTemplate::is(std::vector<int> p)
    {
        for (int i : p)
        { 
            if (std::abs(i) != std::abs(this -> _pdgid)){} 
            else {return true;}
        }
        return false; 
    }
    bool CyParticleTemplate::is_b()  { return this -> is({5}); }
    bool CyParticleTemplate::is_nu() { return this -> is(this -> nudef); }
    bool CyParticleTemplate::is_lep(){ return this -> is(this -> lepdef); }
    bool CyParticleTemplate::is_add()
    { 
        bool out = (this -> is_lep() || this -> is_nu() || this -> is_b()); 
        return !out; 
    }

    bool CyParticleTemplate::lep_decay()
    {
        bool nu  = false; 
        bool lep = false; 
        std::map<std::string, CyParticleTemplate*>::iterator it; 
        for (it = this -> children.begin(); it != this -> children.end(); ++it)
        {
            if (!nu) { nu  = it -> second -> is_nu() ; continue; }
            if (!lep){ lep = it -> second -> is_lep(); continue; }
            if (lep && nu){ return true; }
        }
        return false;
    }

    // Cartesian Conversion functions
    void CyParticleTemplate::ToCartesian() 
    {
        if (!this -> _cartesian){ return; }
        this -> _px = (this -> _pt)*std::cos(this -> _phi); 
        this -> _py = (this -> _pt)*std::sin(this -> _phi); 
        this -> _pz = (this -> _pt)*std::sinh(this -> _eta); 
        this -> _cartesian = false; 
    }
    
    double CyParticleTemplate::px()
    {
        this -> ToCartesian(); 
        return this -> _px;
    }
    double CyParticleTemplate::py()
    {
        this -> ToCartesian(); 
        return this -> _py;
    }
    double CyParticleTemplate::pz()
    {
        this -> ToCartesian(); 
        return this -> _pz;
    }

    // Polar Conversion functions
    void CyParticleTemplate::ToPolar()
    {
        if (!this -> _polar){ return; }

        // Transverse Momenta
        this -> _pt  = std::pow(this -> _px, 2); 
        this -> _pt += std::pow(this -> _py, 2);
        this -> _pt  = std::pow(this -> _pt, 0.5); 

        // Rapidity 
        this -> _eta = std::asinh(this -> _pz/this -> _pt); 
        this -> _phi = std::atan2(this -> _py, this -> _px);  
        this -> _polar = false; 
    }

    double CyParticleTemplate::pt()
    {
        this -> ToPolar(); 
        return this -> _pt;
    }
    double CyParticleTemplate::eta()
    {
        this -> ToPolar(); 
        return this -> _eta;
    }
    double CyParticleTemplate::phi()
    {
        this -> ToPolar(); 
        return this -> _phi;
    }

    void CyParticleTemplate::px(double val)
    { 
        this -> _px = val; 
        this -> _polar = true; 
    }

    void CyParticleTemplate::py(double val)
    { 
        this -> _py = val; 
        this -> _polar = true; 
    }
    
    void CyParticleTemplate::pz(double val)
    { 
        this -> _pz = val; 
        this -> _polar = true; 
    }
    
    // Polar Conversion functions
    void CyParticleTemplate::pt(double val)
    { 
        this -> _pt = val; 
        this -> _cartesian = true;
    }
    
    void CyParticleTemplate::eta(double val)
    { 
        this -> _eta = val; 
        this -> _cartesian = true; 
    }
    
    void CyParticleTemplate::phi(double val)
    { 
        this -> _phi = val; 
        this -> _cartesian = true; 
    }

    std::string CyParticleTemplate::hash()
    {
        if ((this -> _hash).size() != 0)
        {
            return this -> _hash;
        }

        this -> ToCartesian(); 
        this -> _hash  = ToString(this -> _px); 
        this -> _hash += ToString(this -> _py); 
        this -> _hash += ToString(this -> _pz);
        this -> _hash += ToString(this -> _e); 
        this -> _hash  = Hashing(this -> _hash); 
        return this -> _hash; 
    }

    void CyParticleTemplate::addleaf(std::string key, std::string leaf)
    {
        this -> leaves[key] = leaf; 
    }

    CyParticleTemplate* CyParticleTemplate::operator+(CyParticleTemplate* p)
    {
        p -> ToCartesian(); 
        CyParticleTemplate* p2 = new CyParticleTemplate(
                p -> _px + this -> px(), 
                p -> _py + this -> py(), 
                p -> _pz + this -> pz(), 
                p -> _e  + this -> e()
        ); 
        p2 -> ToCartesian(); 
        p2 -> ToPolar(); 
        p2 -> type = this -> type; 
        p2 -> children = this -> children; 
        p2 -> parent = this -> parent; 

        std::map<std::string, CyParticleTemplate*>::iterator it; 
        for (it = p -> children.begin(); it != p -> children.end(); ++it)
        {
            if (p2 -> children.count(it -> first) != 0){continue;}
            p2 -> children[it -> first] = it -> second;
        }

        for (it = p -> parent.begin(); it != p -> parent.end(); ++it)
        {
            if (p2 -> parent.count(it -> first) != 0){continue;}
            p2 -> parent[it -> first] = it -> second;
        }
        return p2; 
    }

    void CyParticleTemplate::operator+=(CyParticleTemplate* p)
    {
        p -> ToCartesian(); 
        this -> ToCartesian();

        this -> _px += p -> _px; 
        this -> _py += p -> _py; 
        this -> _pz += p -> _pz; 
        this -> _e  += p -> _e; 
        
        std::map<std::string, CyParticleTemplate*>::iterator it; 
        for (it = p -> children.begin(); it != p -> children.end(); ++it)
        {
            if (this -> children.count(it -> first) != 0){continue;}
            this -> children[it -> first] = it -> second;
        }

        for (it = p -> parent.begin(); it != p -> parent.end(); ++it)
        {
            if (this -> parent.count(it -> first) != 0){continue;}
            this -> parent[it -> first] = it -> second;
        }
    }

    void CyParticleTemplate::iadd(CyParticleTemplate* p)
    {
        *this += p; 
    }

    bool CyParticleTemplate::operator==(CyParticleTemplate* p)
    {
        return this -> hash() == p -> hash(); 
    }

}
