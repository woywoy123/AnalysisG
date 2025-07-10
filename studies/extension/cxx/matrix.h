#ifndef H_MATRIX 
#define H_MATRIX
#include "particle.h"

double costheta(particle* p1, particle* p2); 
double sintheta(particle* p1, particle* p2);

double** matrix(int row, int col); 
void clear(double** mx, int row, int col); 
void print(double** mx, int prec = 12, int w = 16); 
void print_(double** mx, int row, int col, int prec = 12, int w = 16);

double** unit(); 
double** smatx(double px, double py, double pz); 

double   det(double** v); 
double** inv(double** v); 
double** cof(double** v); 
double** scale(double** v, double s); 
double** arith(double** v1, double** v2, double s = 1); 

double** T(double** v1, int r, int c); 
double** dot(double** v1, double** v2, int r1 = 3, int c1 = 3, int r2 = 3, int c2 = 3); 

double** get_intersection_angle(double** H1, double** H2); 
double** find_roots(double a, double b, double c, double d, double e); 
double** find_roots(double a, double b, double c); 
void intersection_ellipses(double** A, double** B, double eps = 1e-10); 

#endif
