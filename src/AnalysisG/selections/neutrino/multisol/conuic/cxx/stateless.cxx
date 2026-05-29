#include <conuic/stateless.h>
#include <conuic/constants.h>
#include <math.h>

long double omega(long double cos, long double sin, long double r, long double sign){
    return (1.0L / sin) * (sign * r - cos); 
}

long double Omega(long double w, long double beta){
    return std::sqrt(w * w + 1 - beta * beta); 
}

long double Gamma(base_t* pl, base_t* ms, long double sign){
    long double O = branch(sign, pl, ms) -> O; 
    return (pl -> w + sign * ms -> w) / (O * O); 
}

long double delta(base_t* pl, base_t* ms, long double sign){
    long double wpm   = pl -> w * ms -> w; 
    long double Opm   = pl -> O * ms -> O; 
    long double b2_mu = pl -> b_mu * pl -> b_mu; 
    return (1.0L - b2_mu - wpm + sign * Opm) / (pl -> w + ms -> w); 
}

long double KappaPk1(base_t* br, pk1l_t* kx){
    long double ik = 2.0L * kx -> GP * kx -> GM;
    long double fk = br -> w * (kx -> dp + kx -> dm); 
    long double fj = kx -> dpm * (1 + br -> b_mu * br -> b_mu - br -> w * br -> w); 
    ik = (ik * (fk + fj)); 
   
    long double G2p = kx -> GP * kx -> GP;
    long double G2m = kx -> GM * kx -> GM; 

    fk = G2m * kx -> dm + G2p * kx -> dp; 
    fk = br -> O * br -> O * fk - (G2m + G2p) * (kx -> dp + kx -> dm + 2 * br -> w); 
    fk = fk * std::sqrt(-kx -> dpm); 
    return std::atan2(ik, fk) * 0.5L;  
}


