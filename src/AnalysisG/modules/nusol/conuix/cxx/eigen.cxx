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

void Conuix::characteristic::dPdtau_t::PL0(atomics_t* tx){
    auto branch =[](
            long double sign, std::complex<long double> kf,
            std::complex<long double>   u_p, std::complex<long double> u_m, 
            std::complex<long double> alp_p, std::complex<long double> alp_m, 
            std::complex<long double> bta_p, std::complex<long double> bta_m,
            std::complex<long double> out[4]) ->  void 
    {
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

    auto newton =[](
            long double k, int* x,
            long double _a, long double _b, 
            long double _c, long double _d, 
            std::complex<long double> u) -> std::complex<long double>
    {
        std::complex<long double> dq = std::sqrt(1.0L - u * u); 
        std::complex<long double>  N = (_a * u + _b); 
        std::complex<long double>  D = (_d - u * _c); 

        // ------- Mobius ------- //
        std::complex<long double> md = k * dq + N / (D * D); 
        if (std::fabs(md) < 1e-14){*x = 999; return std::atanh(u);}
      
        // ------- dMobius -------- //  
        std::complex<long double> dm = (-k * u / dq) + (_a * D + 2.0 * _c * N) / (D * D * D);
        return u - md / dm; // update u
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
    // ------- Eigenvalues -------- //
    // (beta_mu + Omega +- sqrt[ (beta_mu + Omega)^2 - 4 beta_mu Omega (Omega^2 + beta^2_mu) ])
    // ---------------------------------------------------------------------------------------
    //                          2 sqrt(Omega^2 + beta_mu^2)
    // -- simplify -> l+- = [cos(psi) (beta_mu + Omega) +- sqrt(1 - 2 beta_mu Omega (1 + sin^2(psi)))]/2
    std::complex<long double> dsx = std::sqrt(std::complex<long double>(1.0 - 2.0 * bmu * o * (1.0 + spsi*spsi)));  
    std::complex<long double> lp = (cpsi * (bmu + o) + dsx) * 0.5L; 
    std::complex<long double> lm = (cpsi * (bmu + o) - dsx) * 0.5L; 
    std::complex<long double> kf = lp / lm; 

    // --------- Mobius fixed points ----------- //
    // u+, u-: c u^2 + (d-a) u - b = 0
    // ---> solve quadratc 
    // simplify solutions: 
    // u+: [(Omega - beta_mu) cos(psi) - sqrt([ 1 - 2 Omega beta_mu (1 + sin^2(psi)) ])]/[2 beta_mu sin(psi)]
    // u-: [(Omega - beta_mu) cos(psi) + sqrt([ 1 - 2 Omega beta_mu (1 + sin^2(psi)) ])]/[2 beta_mu sin(psi)]
    std::complex<long double> up = ( (o - bmu)*cpsi - dsx )/ (2.0 * bmu * spsi); // the + - inversion is expected 
    std::complex<long double> un = ( (o - bmu)*cpsi + dsx )/ (2.0 * bmu * spsi); 

    // alpha+-: [(Omega + beta_mu) cos(psi) +- sqrt([....])]/2
    std::complex<long double> ap = ( (o + bmu)*cpsi + dsx ) * 0.5L; 
    std::complex<long double> an = ( (o + bmu)*cpsi - dsx ) * 0.5L; 

    // beta+: [(Omega - beta_mu ) - sqrt([...])] [(Omega + beta_mu ) + sqrt([...])]
    // beta-: [(Omega - beta_mu ) + sqrt([...])] [(Omega + beta_mu ) - sqrt([...])]
    std::complex<long double> bp = ( (o - bmu) * cpsi - dsx ) * ( (o + bmu) * cpsi + dsx )/ (4.0 * bmu * spsi); 
    std::complex<long double> bn = ( (o - bmu) * cpsi + dsx ) * ( (o + bmu) * cpsi - dsx )/ (4.0 * bmu * spsi); 

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
    branch(Kp, kf, up, un, ap, an, bp, bn, gxp); 
    branch(Km, kf, up, un, ap, an, bp, bn, gxn); 

    long double _a =   bmu * cpsi; // beta_mu cos(psi)
    long double _b =   o   * spsi; // Omega   sin(psi)
    long double _c =   bmu * spsi; // beta_mu sin(psi)
    long double _d =   o   * cpsi; // Omega   cos(psi)

    for (int y(0); y < 4; ++y){
        for (int x(0); x < 10; ++x){gxp[y] = newton(Kp, &x, _a, _b, _c, _d, gxp[y]);}
        for (int x(0); x < 10; ++x){gxn[y] = newton(Kp, &x, _a, _b, _c, _d, gxn[y]);}
        this -> tau_sol[2*y] = gxp[y]; 
        this -> solutiond[2*y] = this -> PL0(gxp[y].real()); 

        this -> tau_sol[y*2+1] = gxn[y]; 
        this -> solutiond[y*2+1] = this -> PL0(gxn[y].real()); 
    }
}

void Conuix::characteristic::dPdtau_t::PL1(atomics_t* tx){
    long double _o = tx -> base.o;
    long double _t = tx -> base.w;
    long double _b = tx -> base.beta;
    long double _c = tx -> base.cpsi; 
   
    for (int i(1); i < 5000; ++i){
        // z = e^{2tau}
        long double z = 0.00000001 * i * i; 
        long double tau = 0.5 * std::log(z);   

        long double a_1 = (_o * _t - _b) / (_o * _t + _b); 
        long double b_1 = (_o - _b * _t) / (_o * _t + _b); 
        long double c_1 = (_o + _b * _t) / (_o - _b * _t); 

        long double f1 = std::pow(z + a_1, 2) * std::pow(z + 1, 4);
        long double f2 = -4 * std::pow(_b * _c * _c * _c, 2) * (_o - _b * _t)*(_o * _t + _b); 
        long double f3 = b_1 * b_1 * b_1;
        long double f4 = std::pow(z + c_1, 4) * z; 
        long double prd = f1 + f2 * f3 * f4; 

        long double dpdl0 = this -> L0(1.0L, tau); 
        long double mob   = this -> PL0(tau); 
        long double ply   = tx -> P -> P(dpdl0, 1.0L, tau); 
        std::cout << "tau: " << tau << " Z: " << z << " lambda 0: " << dpdl0 << " char: " << ply << " Mobius: " << mob << " prd: " << prd << std::endl;
//        if (mob > 0.1 ){break;}
    }

    abort(); 


}






void Conuix::characteristic::dPdtau_t::test(atomics_t* tx){
    this -> PL0(tx); 
    this -> PL1(tx); 

    std::cout << "----------------" << std::endl; 
    std::cout << this -> solutiond[0] << " " << this -> tau_sol[0] << std::endl; 
    std::cout << this -> solutiond[1] << " " << this -> tau_sol[1] << std::endl; 
    std::cout << this -> solutiond[2] << " " << this -> tau_sol[2] << std::endl; 
    std::cout << this -> solutiond[3] << " " << this -> tau_sol[3] << std::endl; 
    std::cout << this -> solutiond[4] << " " << this -> tau_sol[4] << std::endl; 
    std::cout << this -> solutiond[5] << " " << this -> tau_sol[5] << std::endl; 
    std::cout << this -> solutiond[6] << " " << this -> tau_sol[6] << std::endl; 
    std::cout << this -> solutiond[7] << " " << this -> tau_sol[7] << std::endl; 




}

