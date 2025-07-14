#ifndef H_TOOLS
#define H_TOOLS

#include "particle.h"
double costheta(particle* p1, particle* p2); 
double sintheta(particle* p1, particle* p2);

void print( double** mx, int prec = 12, int w = 16); 
void print_(double** mx, int row, int col, int prec = 12, int w = 16);

// quadratic 
double** find_roots(double a, double b, double c, int* sx); 

// cubes
double** find_roots(double a, double b, double c, double d, int* sx); 

// quartic
double** find_roots(double a, double b, double c, double d, double e, int* sx); 

// misc
double** unit(); 
double** smatx(double px, double py, double pz); 
double** get_intersection_angle(double** H1, double** H2, double** MET, int* n_sols); 

void multisqrt(double y, double roots[2], int *count); 
void swap_index(double** v, int idx); 

double** make_ellipse(double** H, double angle); 
double distance(double** H1, double a1, double** H2, double a2); 

#endif
