#ifndef H_CONSTANTS_CONUIC
#define H_CONSTANTS_CONUIC

struct kinematic_c;
struct G2_t; 

long double omega(int sign, kinematic_c* data); 
long double Omega(int sign, kinematic_c* data); 

long double Gamma(int sign, kinematic_c* data); 
long double dG2_delta(int sign, kinematic_c* data); 
long double dG2_lambda(int sign, G2_t* data); 
long double asymptote_r(int sign, int rsign, G2_t* data); 


#endif

