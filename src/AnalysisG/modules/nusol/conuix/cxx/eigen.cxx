#include <conuix/solvers.h>
#include <conuix/conuic.h>
#include <iostream>

// NOTE: beta_mu * cosh(tau) * sin(psi) - Omega * sinh(tau) * cos(psi)
long double conuic::gxx(){
    return this -> cache -> gxx_a * this -> ctau + this -> cache -> gxx_b * stau;
}

// NOTE: beta_mu * cosh(tau) * sin(psi) - Omega * sinh(tau) * cos(psi)
long double conuic::gxx(long double _t){
    return this -> cache -> gxx_a * std::cosh(_t) + this -> cache -> gxx_b * std::sinh(_t);
}

// NOTE: Omega * cos(psi) * tanh(tau) - beta_mu * sin(psi)
long double conuic::gtx(){
    return this -> cache -> gtx_a * this -> ctau + this -> cache -> gtx_b;
}

// NOTE: Omega * cos(psi) * tanh(tau) - beta_mu * sin(psi)
long double conuic::gtx(long double _t){
    return this -> cache -> gtx_a * std::tanh(_t) + this -> cache -> gtx_b;
}

// NOTE: Omega * cos(psi) - beta_mu * sin(psi) * tanh(tau)
long double conuic::dtx(){
    return this -> cache -> gtx_a + this -> cache -> gtx_b *  this -> ctau;
}

// NOTE: 
// Omega * cos(psi) - beta_mu * sin(psi) * tanh(tau)
long double conuic::dtx(long double _t){
    return this -> cache -> gtx_a + this -> cache -> gtx_b * std::tanh(_t);
}

// -------------------------- base functions ------------------------ //
long double conuic::P(){
    // P = - l^3 
    //     + Z/O * l^2 
    //     - l * Z^2/O * [beta_mu * sin(psi) * cosh(tau) - O * cos(psi) * sinh(tau)]
    //     - Z^3 / (O * cos(psi)) * sinh(tau) 

    long double _z = this -> scale;
    long double lm = this -> lamb;

    long double a = this -> cache -> p_a;
    long double b = this -> cache -> p_b * _z; 
    long double c = this -> cache -> p_c * _z * _z * this -> gxx();
    long double d = this -> cache -> p_d * _z * _z * _z * this -> stau; 
    return a * lm * lm * lm + b * lm * lm + c * lm + d; 
}

std::complex<long double> conuic::P(std::complex<long double> _l, long double _z, long double _t){
    // P = - l^3 
    //     + Z/O * l^2 
    //     - l * Z^2/O * [beta_mu * sin(psi) * cosh(tau) - O * cos(psi) * sinh(tau)]
    //     - Z^3 / (O * cos(psi)) * sinh(tau) 

    long double a = this -> cache -> p_a;
    long double b = this -> cache -> p_b * _z; 
    long double c = this -> cache -> p_c * _z * _z * this -> gxx(_t);
    long double d = this -> cache -> p_d * _z * _z * _z * this -> stau; 
    return a * _l * _l * _l + b * _l * _l + c * _l + d; 
}


long double conuic::dPdL(){
    long double _z  = this -> scale;
    long double lm = this -> lamb;

    long double a = this -> cache -> dpdl_a * lm * lm;
    long double b = this -> cache -> dpdl_b * lm * _z; 
    long double c = this -> cache -> dpdl_c * _z  * _z * this -> gxx(); 
    return a + b + c;  
}

long double conuic::dPdZ(){
    long double _z  = this -> scale;
    long double lm = this -> lamb;

    long double a = this -> cache -> dpdz_a * lm * lm; 
    long double b = this -> cache -> dpdz_b * lm * _z; 
    long double c = this -> cache -> dpdz_c * _z  * _z * this -> gxx(); 
    return a + b + c;
}

long double conuic::dPdtau(){
    long double _z  = this -> scale;
    long double lm = this -> lamb;
    
    long double a = this -> cache -> dpdt_a * this -> ctau * _z * _z; 
    return a * (lm * this -> gtx() + _z * this -> cache -> dpdt_b); 
}
// -------------------------- base functions ------------------------ //



// -------------------------- special functions ------------------------ //
// NOTE: This is the value of lambda when dP/dZ = 0.
// dP/dZ = a lambda^2 + 2 b Z lambda - 3 c Z^2
// lambda = Z ( alpha +- sqrt( alpha**2 + 3 * sinh(tau) * 1/cos(psi) )
std::complex<long double> conuic::lambda_dPdZ(
        long double _z, long double _t, 
        std::complex<long double>* lp, std::complex<long double>* Pp, 
        std::complex<long double>* lm, std::complex<long double>* Pm
){
    long double alpha = this -> gxx(_t); 
    std::complex<long double> dsc = std::sqrt(alpha * alpha + 3.0 * std::sinh(_t) * (1.0 / this -> cache -> cpsi)); 
    *lp = _z * (alpha + dsc);     *lm = _z * (alpha - dsc); 
    *Pp = this -> P(*lp, _z, _t); *Pm = this -> P(*lm, _z, _t); 
    return dsc;  
}

// NOTE: This is the value of lambda when dP/dL = 0.
// dP/dL = lambda^2 - 2 Z alpha lambda - 3 Z^2 sinh(tau) / cos(psi)
// lambda = (Z/(3 O)) ( 1 +- sqrt( 1 - 3 O alpha )
std::complex<long double> conuic::lambda_dPdL(
        long double _z, long double _t,
        std::complex<long double>* lp, std::complex<long double>* Pp, 
        std::complex<long double>* lm, std::complex<long double>* Pm
){
    std::complex<long double> x = std::complex<long double>(1.0);

    long double sx = _z / (3.0 * this -> cache -> o); 
    std::complex<long double> dsc = std::sqrt(x - 3.0 * this -> cache -> o * this -> gxx(_t)); 
    *lp = sx * (x + dsc);       *lm = sx * (x - dsc); 
    *Pp = this -> P(*lp, _z, _t); *Pm = this -> P(*lm, _z, _t); 
    return dsc;  
}

// NOTE: This is the value of lambda when dP/dtau = 0.
// lambda = Z / (cos(psi) * kappa)
void conuic::lambda_dPdtau(
        long double _z, long double _t,
        std::complex<long double>* lt, std::complex<long double>* Pt
){
    *lt = _z / (this -> cache -> cpsi * this -> dtx(_t)); 
    *Pt = this -> P(*lt, _z, _t); 
}

// -------------------------- special functions ------------------------ //
#include <iomanip>

// NOTE: Omega * sin(psi) + beta_mu * cos(psi) * tanh(tau)
long double conuic::kappa(long double _t, bool use_u){
    long double u = (!use_u) ? std::tanh(_t) : _t;
    return this -> cache -> M_km + this -> cache -> M_kp * u;
}

// NOTE: M(tau) = 
// Omega * cos(psi) - beta_mu * sin(psi) * tanh(tau)
// -------------------------------------------------
// Omega * sin(psi) + beta_mu * cos(psi) * tanh(tau)
long double conuic::Mobius(long double _t, bool use_u, bool check_u){
    long double u = (!use_u) ? std::tanh(_t) : _t;
    long double up = this -> cache -> M_pm - this -> cache -> M_pp * u; 
    long double um = this -> cache -> M_km + this -> cache -> M_kp * u; 
    long double rhs = up/um; 
    if (!check_u){return rhs;}
    rhs = this -> cache -> M_r * rhs * rhs;
    lhs = std::sqrt(1 - u*u) * this -> kappa(_t, use_u); 
    return rhs + 1.0 / lhs; 
}

long double conuic::lambda_dPdtau_qrt(long double _t, bool use_u){
    long double u = (!use_u) ? std::tanh(_t) : _t;
    long double u2 = u * u;
    long double a = this -> cache -> M_qrt.a * u2 * u2; 
    long double a = this -> cache -> M_qrt.a * u2 * u2; 
    long double a = this -> cache -> M_qrt.a * u2 * u2; 
    long double a = this -> cache -> M_qrt.a * u2 * u2; 



}



void conuic::lambda_root_dPdtau(
        long double _z, long double tx,
        std::complex<long double>* lt, std::complex<long double>* Pt,
        bool use_numerical
){













    //long double o    = this -> cache -> o; 
    //long double beta = this -> cache -> beta_l; 
    //long double tpsi = this -> cache -> tpsi; 
    //long double spsi = this -> cache -> spsi;
    //long double cpsi = this -> cache -> cpsi;

    //long double mmu  = this -> cache -> mass_l;
    //long double emu  = this -> cache -> e_l; 


    //long double q2 = std::pow(mmu / emu, 2); 
    //long double t2 = tpsi * tpsi; 

    //long double a =  1;  
    //long double b = -2 * o / (tpsi * beta);
    //long double c = -1 + std::pow(o / (beta * tpsi), 2);  
    //long double d =  2 * o / (tpsi * beta);
    //long double e = std::pow((1 + t2) / (beta * beta * t2), 2) - std::pow( o / (beta * tpsi), 2); 


    //// -------------- get fixed points -------- //
    //std::complex<long double> l1, l2;  

    //long double a_u = beta * cpsi;
    //long double b_u = (beta + o) * spsi; 
    //long double c_u = - o * cpsi; 
    //quadratic(a_u, b_u, c_u, &l1, &l2); 
    //std::complex<long double> u1 = l1;
    //std::complex<long double> u2 = l2; 

    //long double a_m = o * cpsi;
    //long double b_m = beta * spsi;
    //long double c_m = o * spsi;
    //long double d_m = beta * cpsi; 

    //std::complex<long double> fn = (a_m + d_m) * 0.5; 
    //std::complex<long double> ds = std::pow( (a_m - d_m) * 0.5, 2) + b_m * c_m;
    //std::complex<long double> lambda_1 = fn + std::sqrt(ds); 
    //std::complex<long double> lambda_2 = fn - std::sqrt(ds); 
    //std::complex<long double> kf = lambda_1 / lambda_2; 

    //std::complex<long double> one = 1;
    //std::complex<long double> two = 2;
    //std::complex<long double> a_w =         std::pow(beta * cpsi * kf, 2) - one; 
    //std::complex<long double> b_w = two * ( std::pow(beta * cpsi * kf, 2) * u1 * u2 + one);
    //std::complex<long double> c_w =         std::pow(beta * cpsi * kf * u1 * u2, 2) - one; 
    //quadratic(a_w.real(), b_w.real(), c_w.real(), &l1, &l2); 
    //std::complex<long double> zp = l1;
    //std::complex<long double> zm = l2;  

    //long double spx = o * cpsi / ( beta * spsi); //(o * cpsi - beta * spsi * u); 

    //std::cout << "u: " << u1 << " " << u2 << std::endl;
    //std::cout << "l: " << lambda_1 << " " << lambda_2 << std::endl;
    //std::cout << "kf: " << kf << std::endl;
    //std::cout << "a: " << a_w << " b: " << b_w << " c: " << c_w << std::endl; 
    //std::cout << "z+: " << l1 << " z-: " << l2  << std::endl;

    //
    //std::complex<long double> wpp =  std::sqrt(zp); 
    //std::complex<long double> wpm = -std::sqrt(zp); 

    //std::complex<long double> wmp =  one / std::sqrt(zm); 
    //std::complex<long double> wmm = -one / std::sqrt(zm); 
    //std::cout << "w: " << wpp << " " << wpm << " " << wmp << " " << wmm << std::endl; 

    //std::complex<long double> spp = (u1 - wpp * u2)/(one - wpp);    
    //std::complex<long double> spm = (u1 - wpm * u2)/(one - wpm);    
    //std::complex<long double> smp = (u1 - wmp * u2)/(one - wmp);    
    //std::complex<long double> smm = (u1 - wmm * u2)/(one - wmm);    
    //std::cout << "u: " << spp << " " << spm << " " << smp << " " << smm << std::endl; 
    //
    ////std::complex<long double> sxl = (std::pow(beta * cpsi * kf, 2) - one) * zp * zp;
    ////sxl += two * (std::pow(beta * cpsi * kf, 2) * u1 * u2 + one) * zp; 
    ////sxl += (std::pow(beta * cpsi * kf * u1 * u2, 2) - one); 
    ////std::cout << sxl << std::endl;


    //long double mob = this -> Mobius(spp.real(), true);
    //long double ssx = (spsi * spsi - beta * beta * cpsi * cpsi) * std::pow(mob, 4);
    //ssx += -2 * spsi * cpsi * std::pow(mob, 3);
    //ssx += cpsi * cpsi * (2 - beta*beta) * std::pow(mob, 2);
    //ssx +=  2 * spsi * cpsi * mob + spsi * spsi; 
    //std::cout << ssx << std::endl;

    ////int lx = 1000000; 
    ////for (int x(0); x < lx; ++x){
    ////    long double _t = tx + (lx/2 - x)*0.00001; 

    ////    long double u   = std::tanh(_t);
    ////    long double rhs = std::cosh(_t) / this -> kappa(_t); 
    ////    long double lhs = this -> Mobius(_t);

    ////    long double fhs = (cpsi * beta);
    ////    long double dlt = fhs * lhs * lhs + rhs; 
    ////    std::cout << _t << " " << dlt << " " << rhs << " " << mob << " " << rs << " " << std::endl;


    ////}
    ////abort(); 



}
