#include <conuic/constants.h>
#include <conuic/stateless.h>
#include <conuic/atomics.h>
#include <math.h>

shared_t::shared_t(kinematics_t* bq, kinematics_t* lp){
    this -> cos = costheta(bq, lp); 
    this -> sin = cs_sin(this -> cos); 
    this -> tan = cs_tan(this -> cos); 
    this -> theta = std::acos(this -> cos);  

    this -> r = lp -> b / bq -> b; 
    this -> m_mu = lp -> m; 
    this -> b_mu = lp -> b; 
    this -> e_mu = lp -> e;
    this -> p_mu = lp -> p;

    this -> m_bq = bq -> m; 
    this -> b_bq = bq -> b; 
    this -> e_bq = bq -> e;
    this -> p_bq = bq -> p; 
}

shared_t::~shared_t(){}

base_t::base_t(shared_t* shr, long double sign){
    this -> w = omega(shr -> cos, shr -> sin, shr -> r, sign); 
    this -> O = Omega(this -> w, shr -> b_mu); 
    this -> b_mu = shr -> b_mu; 
    this -> e_mu = shr -> e_mu; 

    this -> A = (pw(shr -> b_mu) - pw(this -> w)) / pw(this -> O); 
    this -> B = 2 * this -> w / pw(this -> O); 
    this -> C = - (1.0L - pw(shr -> b_mu)) / pw(this -> O); 
    this -> D = 2 * shr -> p_mu; 
    this -> E = pw(shr -> m_mu); 

    this -> track("w", &this -> w); 
    this -> track("O", &this -> O); 
}

base_t::~base_t(){}

pk1l_t::pk1l_t(base_t* pl, base_t* ms){
    this -> GP = Gamma(pl, ms, +1); 
    this -> GM = Gamma(pl, ms, -1); 

    this -> dp = delta(pl, ms, +1); 
    this -> dm = delta(pl, ms, -1); 
    this -> dpm = this -> dp * this -> dm; 

    this -> gmu = std::sqrt(1 - pl -> b_mu * pl -> b_mu); 
    this -> eta = std::atanh(this -> dm / this -> gmu); 
    
    this -> kap = KappaPk1(pl, this); 
    this -> kam = KappaPk1(ms, this); 

    long double fx = pl -> e_mu / pl -> b_mu; 

    this -> L0pp = fx * this -> GP * std::sinh(this -> eta) * (this -> dp * pl -> w + this -> dpm); 
    this -> L0pm = fx * this -> GM * std::cosh(this -> eta) * (this -> dm * pl -> w + this -> dpm); 

    this -> L0mp = fx * this -> GP * std::sinh(this -> eta) * (this -> dp * ms -> w + this -> dpm); 
    this -> L0mm = fx * this -> GM * std::cosh(this -> eta) * (this -> dm * ms -> w + this -> dpm); 

}

long double pk1l_t::lx(long double sx, long double sy){
    return sx - this -> dp * sy;
}

long double pk1l_t::ly(long double sx, long double sy){
    return sx - this -> dm * sy;
}

long double pk1l_t::sx(long double _lx, long double _ly){
    return (this -> dp * _ly - this -> dm * _lx) / (this -> dp - this -> dm);
}

long double pk1l_t::sy(long double _lx, long double _ly){
    return (_ly - _lx) / (this -> dp - this -> dm);
}

long double pk1l_t::Lx(long double _sx, long double _sy){
    return this -> GP * (std::sinh(this -> eta) * _sx + this -> gmu * std::cosh(this -> eta) * _sy);
}

long double pk1l_t::Ly(long double _sx, long double _sy){
    return this -> GM * (std::cosh(this -> eta) * _sx - this -> gmu * std::sinh(this -> eta) * _sy);
}

long double pk1l_t::Sx(long double _lx, long double _ly){
    _lx = _lx / this -> GP; _ly = _ly / this -> GM; 
    long double s = std::sinh(this -> eta) * _lx + std::cosh(this -> eta) * _ly;
    return s / std::cosh(2 * this -> eta); 
}

long double pk1l_t::Sy(long double _lx, long double _ly){
    _lx = _lx / this -> GP; _ly = _ly / this -> GM; 
    long double s = std::cosh(this -> eta) * _lx - std::sinh(this -> eta) * _ly;
    return s / (this -> gmu * std::cosh(2 * this -> eta)); 
}

pk1l_t::~pk1l_t(){}
