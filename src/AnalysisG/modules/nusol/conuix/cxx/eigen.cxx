#include <conuix/solvers.h>
#include <conuix/conuic.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>

long double Conuix::characteristic::poly_t::xlinear(long double tau){
    return this -> x_a * std::sinh(tau) + this -> x_b * std::cosh(tau);
}

long double Conuix::characteristic::poly_t::ylinear(long double tau){
    return this -> y_a * std::sinh(tau) + this -> y_b * std::cosh(tau) ;
}

long double Conuix::characteristic::poly_t::xyratio(long double tau){
    return this -> xlinear(tau) / this -> ylinear(tau);
}

long double Conuix::characteristic::poly_t::P(long double lambda, long double Z, long double tau){return 0.0L;}


Conuix::characteristic::P_t::P_t(Conuix::base_t* base){
    this -> a = - 1;
    this -> b =   1.0 / base -> o;
    this -> c = - 1.0 / base -> o;
    this -> d = - 1.0 / (base -> cpsi * base -> o);  

    // using factor: [beta_mu sin(psi) cosh(tau) - Omega cos(psi) sinh(tau)]
    this -> y_b =   base -> beta * base -> spsi;
    this -> y_a = - base -> o    * base -> cpsi; 
}

long double Conuix::characteristic::P_t::P(long double lambda, long double Z, long double tau){
    // P = - l^3 
    //     + Z/O * l^2 
    //     - l * Z^2/O * [beta_mu * sin(psi) * cosh(tau) - O * cos(psi) * sinh(tau)]
    //     - Z^3 / (O * cos(psi)) * sinh(tau) 
    long double x1 = lambda * lambda * lambda;
    long double x2 = lambda * lambda * Z; 
    long double x3 = lambda * Z * Z * this -> ylinear(tau);
    long double x4 = Z * Z * Z * std::sinh(tau);

    return x1 * this -> a + x2 * this -> b + x3 * this -> c + x4 * this -> d;  
}

void Conuix::characteristic::P_t::print(int p){
    this -> prec = p; 
    this -> variable("P(lambda)", this -> P(1, 1, 0)); 
}











Conuix::characteristic::dPdtau_t::dPdtau_t(Conuix::base_t* base){
    this -> a = - 1.0 / base -> o;
    this -> b = - 1.0 / (base -> cpsi * base -> o);
    this -> c = - 1.0 / base -> cpsi; 
    this -> cf = (base -> cpsi * base -> cpsi) * base -> beta; 

    // using factor: [beta_mu sin(psi) sinh(tau) - Omega cos(psi) cosh(tau)]
    this -> x_a =   base -> beta * base -> spsi;
    this -> x_b = - base -> o    * base -> cpsi; 

    // using factor: [beta_mu cos(psi) sinh(tau) + Omega sin(psi) cosh(tau)]
    this -> y_a = base -> beta * base -> cpsi;
    this -> y_b = base -> o    * base -> spsi; 
}

long double Conuix::characteristic::dPdtau_t::P(long double lambda, long double Z, long double tau){
    // dP/dtau = - Z^2 lambda/Omega [beta_mu sin(psi) sinh(tau) - Omega cos(psi) cosh(tau)] 
    //           - (Z^3 / Omega)(cosh(tau)/cos(psi))
    long double x1 = Z * Z * lambda * this -> xlinear(tau);
    long double x2 = Z * Z * Z * std::cosh(tau); 
    return x1 * this -> a + x2 * this -> b;
}

long double Conuix::characteristic::dPdtau_t::L0(long double Z, long double tau){
    // lambda* = - Z cosh(tau)/(cos(psi) * [beta_mu sin(psi) sinh(tau) - Omega cos(psi) cosh(tau)])    
    return Z * this -> c * std::cosh(tau) / this -> xlinear(tau);
}

long double Conuix::characteristic::dPdtau_t::PL0(long double tau){
    // 0 = cosh(tau)^2/[Omega sin(psi) cosh(tau) + beta_mu cos(psi) sinh(tau)] 
    //   + beta_mu cos(psi)^2 (
    //          [beta_mu sin(psi) sinh(tau) - Omega cos(psi) cosh(tau)]
    //          =======================================================
    //          [beta_mu cos(psi) sinh(tau) + Omega sin(psi) cosh(tau)]
    //  )^2
    long double lhs = std::cosh(tau);
    long double rhs = this -> xyratio(tau);
    return (lhs * lhs)/this -> ylinear(tau)  + rhs * rhs * this -> cf;
}

long double Conuix::characteristic::dPdtau_t::PL0(atomics_t* tx){
    auto branch =[](
            long double sign, std::complex<long double> kf,
            std::complex<long double>   u_p, std::complex<long double> u_m, 
            std::complex<long double> alp_m, std::complex<long double> alp_p, 
            std::complex<long double> bta_m, std::complex<long double> bta_p,
            std::complex<long double> out[4]
    ) ->  void {
        const long double k2 = sign*sign;

        std::complex<long double> p4 =         k2 * std::pow(alp_m, 4); 
        std::complex<long double> p3 = -4.0L * k2 * std::pow(alp_m, 3) * alp_p; 
        std::complex<long double> p2 =  6.0L * k2 * std::pow(alp_m * alp_p, 2) - 2.0L * bta_m * bta_m; 
        
        // palindromic divide by z^2
        // w = z + lambda/z
        // w^2 - (p3/p4) w + (p2 / p4 - 2lambda) = 0
        std::complex<long double> aw = 1.0L; 
        std::complex<long double> bw = - (p3/p4);
        std::complex<long double> cw = (p2/p4) - 2.0L * kf; 
        std::complex<long double> sx = std::sqrt(bw * bw - 4.0L * aw * cw); 

        // solve: z^2 - w z + lambda = 0
        std::complex<long double>   w1 = (-bw + sx) / (2.0L * aw); 
        std::complex<long double>  _w1 = std::sqrt(w1 * w1 - 4.0L * kf); 
        std::complex<long double> z1w1 = (w1 + _w1) * 0.5L; 
        std::complex<long double> z2w1 = (w1 - _w1) * 0.5L; 
        out[0] = (u_p - u_m * z1w1) / (1.0L - z1w1); 
        out[1] = (u_p - u_m * z2w1) / (1.0L - z2w1); 

        std::complex<long double>   w2 = (-bw - sx) / (2.0L * aw); 
        std::complex<long double>  _w2 = std::sqrt(w2 * w2 - 4.0L * kf); 
        std::complex<long double> z1w2 = (w2 + _w2) * 0.5L; 
        std::complex<long double> z2w2 = (w2 - _w2) * 0.5L; 
        out[2] = (u_p - u_m * z1w2) / (1.0L - z1w2); 
        out[3] = (u_p - u_m * z2w2) / (1.0L - z2w2); 
    }; 


    // ------- constants ---------- //
    long double spsi = tx -> base.spsi; 
    long double cpsi = tx -> base.cpsi; 
    long double bmu  = tx -> base.beta; 
    long double o    = tx -> base.o;
    
    // ------- Mobius local variables ------ //
    // M(u) = 
    //      [a   b] [u+]
    //      [c   d] [u-]
    //
    //long double _a =   bmu * cpsi; // beta_mu cos(psi)
    //long double _b =   o   * spsi; // Omega   sin(psi)
    //long double _c = - bmu * spsi; //-beta_mu sin(psi)
    //long double _d =   o   * cpsi; // Omega   cos(psi)

    // ------- Eigenvalues -------- //
    // (beta_mu + Omega +- sqrt[ (beta_mu + Omega)^2 - 4 beta_mu Omega (Omega^2 + beta^2_mu) ])
    // ---------------------------------------------------------------------------------------
    //                          2 sqrt(Omega^2 + beta_mu^2)
    // -- simplify -> l+- = [cos(psi) (beta_mu + Omega) +- sqrt(1 - 2 beta_mu Omega (1 + sin^2(psi)))]/2
    std::complex<long double> dsx = std::sqrt(1.0L - 2.0L * bmu * o * (1.0L + spsi*spsi));  
    std::complex<long double> lp = (cpsi * (bmu + o) + dsx) * 0.5L; 
    std::complex<long double> lm = (cpsi * (bmu + o) - dsx) * 0.5L; 
    std::complex<long double> kf = lp / lm; 

    // --------- Mobius fixed points ----------- //
    // u+, u-: c u^2 + (d-a) u - b = 0
    // ---> solve quadratc 
    // simplify solutions: 
    // u+: [(Omega - beta_mu) cos(psi) - sqrt([ 1 - 2 Omega beta_mu (1 + sin^2(psi)) ])]/[2 beta_mu sin(psi)]
    // u-: [(Omega - beta_mu) cos(psi) + sqrt([ 1 - 2 Omega beta_mu (1 + sin^2(psi)) ])]/[2 beta_mu sin(psi)]
    std::complex<long double> dxs = std::sqrt(1.0L - 2.0L * o * bmu * ( 1.0L + spsi * spsi ));
    std::complex<long double> up = ( (o + bmu)*cpsi - dxs ) / (2.0L * bmu * spsi); // the + - inversion is expected 
    std::complex<long double> un = ( (o + bmu)*cpsi + dxs ) / (2.0L * bmu * spsi); 
    
    // alpha+-: [(Omega + beta_mu) cos(psi) +- sqrt([....])]/2
    std::complex<long double> ap = ( (o + bmu)*cpsi + dxs ) * 0.5L; 
    std::complex<long double> an = ( (o + bmu)*cpsi + dxs ) * 0.5L; 

    // beta+-: [(Omega - beta_mu ) + (Omega + beta_mu) sin(psi)^2 +- cos(psi) sqrt([....])]/(2 sin(psi))
    std::complex<long double> bp = ( (o - bmu) + (o + bmu) * spsi * spsi - dxs * cpsi ) / (2.0L * spsi); 
    std::complex<long double> bn = ( (o - bmu) + (o + bmu) * spsi * spsi + dxs * cpsi ) / (2.0L * spsi); 

    // ----------------- More explanation of what all this means ----------------------- //
    // Mobius transforms have fixed points - u+/u-
    // But we need them in their eigenbasis to simplify the complicated sextic we would have to solve
    // So we use the property of Mobius transforms:
    // u+(-) = [b + a u+(-) ]/[ d + c u+(-) ] = beta+(-)/alpha+(-)
    // To effectively "estimate" initial solutions of the sextic by 
    // approximating a sextic as a (quadratic) x (Quartic)

    long double Kp =  bmu * cpsi * cpsi;
    long double Km = -bmu * cpsi * cpsi;

    std::complex<long double> gxp[4], gxn[4];
    branch(Kp, kf, up, un, ap, an, bn, bp, gxp); 
    branch(Km, kf, up, un, ap, an, bn, bp, gxn); 
  
    std::cout << "---------" << std::endl; 
    std::cout << gxp[0] << std::endl;
    std::cout << gxp[1] << std::endl;
    std::cout << gxp[2] << std::endl;
    std::cout << gxp[3] << std::endl;

    std::cout << gxn[0] << std::endl;
    std::cout << gxn[1] << std::endl;
    std::cout << gxn[2] << std::endl;
    std::cout << gxn[3] << std::endl;




    return 0.0L; 
}

void Conuix::characteristic::dPdtau_t::test(atomics_t* tx){
    this -> PL0(tx); 





    abort(); 

}

