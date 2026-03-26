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

long double m_nuG(delta_t* dt, branches_t* br, kinematics_t* kl, long double tau, long double phi, int sign, int eps); 
// see https://mathcurve.com/surfaces.gb/hyperboloid/hyperboloid2.shtml <<---- lower image [2026]










#endif 

