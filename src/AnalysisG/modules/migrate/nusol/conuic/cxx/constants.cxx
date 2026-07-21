#include <conuic/constants.h>
#include <conuic/variables.h>
#include <conuic/factor.h>

long double omega(int sign, kinematic_c* data){
    long double s = signs(sign, 1.0L, -1.0L); 
    return (1 / data -> theta.sin) * (s * data -> b_mu / data -> b_b - data -> theta.cos); 
}

long double Omega(int sign, kinematic_c* data){
    long double w = data -> w.pair(sign); 
    return std::sqrt(w * w + 1 - data -> b_mu * data -> b_mu); 
}

long double dG2_delta(int sign, kinematic_c* data){
    long double s   = signs(sign, 1.0L, -1.0L); 
    long double Opm = data -> O.p * data -> O.m; 
    long double wpm = data -> w.p * data -> w.m; 
    long double spm = data -> w.p + data -> w.m; 
    long double b2  = data -> b_mu * data -> b_mu;
    return (1 - b2 - wpm + s * Opm)/spm;  
}

long double Gamma(int sign, kinematic_c* data){
    long double o = data -> O.pair(sign); 
    long double s = signs(sign, 1.0L, -1.0L); 
    return (data -> w.p + s * data -> w.m) / (o * o); 
}

long double dG2_lambda(int sign, G2_t* data){
    long double s = signs(sign, 1.0L, -1.0L); 
    long double a = data -> G.p * data -> G.m / (2 * data -> kappa.pA.cos * data -> kappa.mA.cos); 
    return - a * (data -> psi.mA.cos - s); 
}

long double asymptote_r(int sign, int rsign, G2_t* data){
     long double r_ = data -> w.pair(sign) + signs(rsign, 1.0L, -1.0L) * data -> b_mu * data -> O.pair(sign); 
     return r_ / (1 - data -> b2_mu); 
}

long double detG2(G2_t* data){
    long double angle = std::sin(data -> kappa.pA.phi - data -> kappa.mA.phi); 
    long double f = - std::pow(data -> G.p * data -> G.m * 0.5, 2) / ( 2 * angle); 

}


