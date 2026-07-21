#ifndef H_BASE_CONUIC
#define H_BASE_CONUIC
#include <complex>

struct kinematic_c;
struct branches_t; 

long double to_sx(long double x1, long double y1, long double delta); 
long double to_sy(long double x1, long double y1, long double delta); 

long double to_x1(int sign, long double sx, long double sy, G2_t* data); 
long double to_y1(int sign, long double sx, long double sy, G2_t* data); 

long double line_sy(int sign, long double sx, kinematic_c* data);
long double line_sx(int sign, long double sy, kinematic_c* data);

long double dG2(long double sx, long double sy, kinematic_c* data); 
long double mobius(long double sx, long double sy, G2_t* data); 

std::complex<long double> mW(long double sx, long double m_nu, kinematic_c* data);
std::complex<long double> mT(long double sx, long double sy, long double m_nu, kinematic_c* data); 
std::complex<long double> mN(int sign, G2_t* data); 

#endif
