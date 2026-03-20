#ifndef MULTISOL_CONUIC_ATOMICS_H
#define MULTISOL_CONUIC_ATOMICS_H

#include <common/matrix.h>
struct kinematics_t; 
struct branches_t;
struct angular_t; 
struct hyper_t; 
struct delta_t;
class conuic; 

long double convert(int v); 
long double convert(double v); 

template <typename g>
g* route(g* o1, g* o2, int sign){return (sign > 0) ? o1 : o2;}

long double mag2(kinematics_t* v1, kinematics_t* v2); 
long double costh(kinematics_t* v1, kinematics_t* v2); 
long double tn_cos(long double tn); 
long double tn_sin(long double tn); 

long double omega(kinematics_t* jx, kinematics_t* lx, int sign);
long double Omega(kinematics_t* jx, kinematics_t* lx, int sign);
long double Gamma(branches_t* plus, branches_t* minus, int sign); 
long double delta(branches_t* plus, branches_t* minus, int sign); 
long double lm_dt(delta_t* dt, int sign);

template <typename g>
void flush(g** sx){
    if (!*sx){return;}
    delete *sx; *sx = nullptr; 
}; 

// SIGMA[kappa, sign]: cos(kappa) - delta[sign] sin(kappa)
long double SigmasE(delta_t* dt, branches_t* br, int sign); // sign here chooses the delta root.

// LAMBDA[kappa, sign]: sin(kappa) + delta[sign] cos(kappa)
long double LambdaE(delta_t* dt, branches_t* br, int sign); // sign here chooses the delta roots. 


// -------- special cases -------- //
// - Explanation: This is the case where the two delta roots lines Sx - delta^+ Sy, Sx - detla^- Sy
// have the same neutrino mass. This is justified by the fact that ALL branches must fundamentally 
// agree on the same invariants. The output is cos(phi) tanh(tau).
// This is certainly not a pretty equation....
long double m_nueq_line(
        delta_t* dt, branches_t* pls, branches_t* msn, kinematics_t* kl, int eps, bool swp
); 
// what does swp mean? It means that the permutation of the delta roots should be symmetric, 
// because any asymmetry indicates there is a twist or relative tilt between the two hyperbolic sheets
// which were introduced due to lorentz distortion, the idea here is that first one finds this exact asymmetry (if present), if there is a twist, then it is correct using cos(phi) anything relating to relative tilt is due to 
// tau, i.e. boost.
// see https://mathcurve.com/surfaces.gb/hyperboloid/hyperboloid2.shtml <<---- lower image [2026]

long double m_nuG(delta_t* dt, branches_t* br, kinematics_t* kl, long double tau, long double phi, int sign, int eps); 




#endif 

