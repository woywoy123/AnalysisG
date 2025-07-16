#ifndef H_TOOLS
#define H_TOOLS
#include "particle.h"

class mtx; 

double costheta(particle* p1, particle* p2); 
double sintheta(particle* p1, particle* p2);

// quadratic 
double** find_roots(double a, double b, double c, int* sx); 

// cubes
double** find_roots(double a, double b, double c, double d, int* sx); 

// quartic
double** find_roots(double a, double b, double c, double d, double e, int* sx); 

// misc
mtx* unit(); 
mtx* smatx(double px, double py, double pz); 
mtx* get_intersection_angle(mtx* H1, mtx* H2, mtx* MET, int* n_sols); 


mtx make_ellipse(mtx* H, double angle); 
double distance(mtx* H1, double a1, mtx* H2, double a2); 

#endif
