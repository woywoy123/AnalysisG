#ifndef H_MULTISOL_SOLVERS
#define H_MULTISOL_SOLVERS
#include <complex.h>
#include <cmath>
struct matrix_t; 

struct roots_t {
    std::complex<long double> a = 0;
    std::complex<long double> b = 0;
    std::complex<long double> c = 0; 
    std::complex<long double> d = 0;
    int num_r = 0; 
    matrix_t vec(); 
}; 

long double det3(long double** data); 
long double det2(long double** data); 

matrix_t circle(); 
matrix_t identity(); 
matrix_t S0(long double metx, long double mety); 
matrix_t V4(long double metx, long double mety, long double metz = 0); 

roots_t find_roots(long double a, long double b, long double c, long double tol); 
roots_t find_roots(long double a, long double b, long double c, long double d, long double tol); 

void factor_degenerate(const matrix_t* G, matrix_t* lines, int* lc); 
int  intersections_ellipse_line(matrix_t* ellipse, matrix_t* line, int k, matrix_t* pts); 
matrix_t* intersection_ellipses(matrix_t* A, matrix_t* B); 

#endif
