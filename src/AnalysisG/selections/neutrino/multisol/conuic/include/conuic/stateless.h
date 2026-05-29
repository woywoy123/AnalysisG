#ifndef MULTISOL_CONUIC_STATELESS_H
#define MULTISOL_CONUIC_STATELESS_H

struct base_t; 
struct pk1l_t; 
struct shared_t; 

template <typename g>
g* branch(long double sign, g* pl, g* ms){return (sign > 0) ? pl : ms;}; 

long double omega(long double cos, long double sin, long double r, long double sign);
long double Omega(long double w, long double beta); 

long double Gamma(base_t* pl, base_t* ms, long double sign); 
long double delta(base_t* pl, base_t* ms, long double sign); 

long double KappaPk1(base_t* br, pk1l_t* kx); 


#endif 

