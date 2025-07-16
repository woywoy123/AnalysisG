#ifndef H_MATRIX 
#define H_MATRIX
#include "tools.h"

class mtx; 
void _copy(double* dst, double* src, int lx); 
double** _copy(double** dst, double** src, int lx, int ly, bool del = false); 

void clear(double** mx, int row, int col); 
double** matrix(int row, int col); 

double** cof(double** v); 
double** scale(double** v, double s); 
double** diag(double** v, int idx); 
double** scale(double** v, int idx, int idy, double s); 
double** arith(double** v1, double** v2, double s = 1, int idx = 3, int idy = 3); 

double** Rxyz(double** M, double alpha, double beta, double gamma); 

double** T(double** v1, int r = 3, int c = 3); 
double** dot(double** v1, double** v2, int r1 = 3, int c1 = 3, int r2 = 3, int c2 = 3); 
double** dot(double** v1, double** v2, bool del, int r1 = 3, int c1 = 3, int r2 = 3, int c2 = 3);

int intersection_ellipses(mtx* A, mtx* B, mtx** lines, mtx** pts, mtx** sols); 


#endif
